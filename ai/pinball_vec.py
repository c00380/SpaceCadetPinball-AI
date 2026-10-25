import multiprocessing as mp
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from game_client import GameClient
from pinball_env import OBS_DIM, PinballEnv


class PinballVecEnv(VecEnv):
    def __init__(self, num_envs, stage="hit"):
        self.envs = [PinballEnv(stage=stage, headless=True) for _ in range(num_envs)]
        super().__init__(num_envs, self.envs[0].observation_space, self.envs[0].action_space)
        with ThreadPoolExecutor(max_workers=min(num_envs, 8)) as pool:
            clients = list(pool.map(lambda _: GameClient(headless=True), range(num_envs)))
        for env, client in zip(self.envs, clients):
            env.start_game(client)
        self._pending = []

    def _reset_env(self, env):
        for attempt in range(3):
            try:
                if env.game is None:
                    env.start_game()
                return env.begin(env.game.reset())
            except (ConnectionError, OSError, RuntimeError):
                env.game = None
                if attempt == 2:
                    raise

    def reset(self):
        self._reset_infos = [{} for _ in self.envs]
        return np.stack([self._reset_env(env) for env in self.envs])

    def step_async(self, actions):
        self._pending = []
        for env, action in zip(self.envs, actions):
            pressed, reward = env.prepare(action)
            try:
                env.game.send_step(pressed[0], pressed[1], env.frame_skip)
                self._pending.append((pressed, reward, True))
            except (ConnectionError, OSError, AttributeError):
                self._pending.append((pressed, reward, False))

    def step_wait(self):
        n = self.num_envs
        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        infos = [{} for _ in range(n)]
        for i, (env, (pressed, reward, sent)) in enumerate(zip(self.envs, self._pending)):
            try:
                if not sent:
                    raise ConnectionError("send failed")
                o, r, terminated, truncated, info = env.finish(env.game.recv(), pressed, reward)
            except (ConnectionError, OSError):
                env.game = None
                o, r, terminated, truncated, info = np.zeros(OBS_DIM, np.float32), 0.0, False, True, {}
            if terminated or truncated:
                info["terminal_observation"] = o
                if truncated and not terminated:
                    info["TimeLimit.truncated"] = True
                o = self._reset_env(env)
                dones[i] = True
            obs[i], rewards[i], infos[i] = o, r, info
        return obs, rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.envs[i], attr_name) for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._get_indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def _worker_main(conn, num_envs, stage):
    try:
        venv = PinballVecEnv(num_envs, stage)
        conn.send(("ok", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            cmd, data = conn.recv()
        except EOFError:
            break
        try:
            if cmd == "step":
                venv.step_async(data)
                conn.send(("ok", venv.step_wait()))
            elif cmd == "reset":
                conn.send(("ok", venv.reset()))
            elif cmd == "env_method":
                name, args, kwargs = data
                conn.send(("ok", venv.env_method(name, *args, **kwargs)))
            elif cmd == "get_attr":
                conn.send(("ok", venv.get_attr(data)))
            elif cmd == "set_attr":
                venv.set_attr(*data)
                conn.send(("ok", None))
            elif cmd == "close":
                venv.close()
                conn.send(("ok", None))
                break
        except Exception:
            conn.send(("error", traceback.format_exc()))
    conn.close()


class PinballParallelVecEnv(VecEnv):
    def __init__(self, num_envs, stage="hit", workers=4):
        self.closed = False
        workers = max(1, min(workers, num_envs))
        self.sizes = [num_envs // workers + (1 if i < num_envs % workers else 0) for i in range(workers)]
        context = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for size in self.sizes:
            parent, child = context.Pipe()
            process = context.Process(target=_worker_main, args=(child, size, stage), daemon=True)
            process.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(process)
        self._gather()
        spaces = PinballEnv(stage=stage)
        super().__init__(num_envs, spaces.observation_space, spaces.action_space)

    def _gather(self):
        results = []
        for conn in self.conns:
            status, payload = conn.recv()
            if status != "ok":
                self.close()
                raise RuntimeError("worker failed:\n" + payload)
            results.append(payload)
        return results

    def _call(self, cmd, data=None):
        for conn in self.conns:
            conn.send((cmd, data))
        return self._gather()

    def reset(self):
        self._reset_infos = [{} for _ in range(self.num_envs)]
        return np.concatenate(self._call("reset"))

    def step_async(self, actions):
        start = 0
        for conn, size in zip(self.conns, self.sizes):
            conn.send(("step", actions[start:start + size]))
            start += size

    def step_wait(self):
        parts = self._gather()
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                np.concatenate([p[2] for p in parts]), [info for p in parts for info in p[3]])

    def close(self):
        if self.closed:
            return
        self.closed = True
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except OSError:
                pass
        for process in self.procs:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [r for part in self._call("env_method", (method_name, method_args, method_kwargs)) for r in part]

    def get_attr(self, attr_name, indices=None):
        return [r for part in self._call("get_attr", attr_name) for r in part]

    def set_attr(self, attr_name, value, indices=None):
        self._call("set_attr", (attr_name, value))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
