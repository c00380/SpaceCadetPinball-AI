import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rewards as R
from game_client import GameClient

MAX_GAME_SECONDS = 1500
N_MISSION = 20
N_LANES = 17
OBS_DIM = 15 + 4 + 8 + 11 + N_MISSION + N_LANES + 2

BALL_SCALE = np.array([1, 1 / 8, 1 / 12, 1 / 40, 1 / 40], dtype=np.float64)


class Indices:

    def __init__(self, names):
        i = {n: k for k, n in enumerate(names)}
        self.balls = np.array([[i[f"b{k}_{f}"] for f in ("act", "x", "y", "vx", "vy")] for k in range(3)])
        self.pivots = np.array([i["lflip_px"], i["lflip_py"], i["rflip_px"], i["rflip_py"]])
        self.flip = np.array([i["lflip_angle"], i["lflip_active"], i["rflip_angle"], i["rflip_active"]])
        self.game = np.array([i[n] for n in ("ball_count", "multiball", "extra_balls", "shoot_again", "ball_save",
                                             "plunger_ready", "score_mult", "mission_ready", "fuel", "tilt")])
        self.lanes = np.array([i[f"lane{k}"] for k in range(N_LANES)])
        self.score, self.mission = i["score"], i["mission"]
        self.time, self.game_over, self.n_balls = i["time"], i["game_over"], i["n_balls"]
        self.drains, self.launches = i["drains"], i["launches"]
        self.hits_l_act, self.hits_r_act = i["hits_l_act"], i["hits_r_act"]
        self.hits_l_pas, self.hits_r_pas = i["hits_l_pas"], i["hits_r_pas"]


class PinballEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage="hit", headless=True, frame_skip=R.FRAME_SKIP):
        super().__init__()
        self.stage = stage
        self.headless = headless
        self.frame_skip = frame_skip
        self.action_space = spaces.MultiDiscrete([2, 2])
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.game = None
        self.raw = None
        self.I = None
        self.ix = None
        self._init_flipper_state()

    def _init_flipper_state(self):
        self.hold = [0, 0]
        self.cooldown = [0, 0]
        self.cmd = [0, 0]
        self.cmd_age = [99, 99]

    def set_stage(self, stage):
        self.stage = stage

    def start_game(self, client=None):
        if self.game is not None:
            self.game.close()
        self.game = client or GameClient(headless=self.headless)
        self.I = self.game.index
        self.ix = Indices(self.game.names)

    def _lane_count(self):
        return float(self.raw[self.ix.lanes[:8]].sum())

    def _obs(self):
        r, ix = self.raw, self.ix
        o = np.empty(OBS_DIM, dtype=np.float32)
        B = r[ix.balls]
        o[0:15] = (B * BALL_SCALE).ravel()
        act = B[:, 0] > 0.5
        if act.any():
            k = int(np.argmax(np.where(act, B[:, 2], -np.inf)))
            bx, by = B[k, 1], B[k, 2]
            P = r[ix.pivots]
            o[15:19] = ((bx - P[0]) / 4, (by - P[1]) / 4, (bx - P[2]) / 4, (by - P[3]) / 4)
        else:
            o[15:19] = 0.0
        F = r[ix.flip]
        o[19:23] = (F[0], F[1], self.hold[0] / R.HOLD_MAX_STEPS, 1.0 if self.cooldown[0] > 0 else 0.0)
        o[23:27] = (F[2], F[3], self.hold[1] / R.HOLD_MAX_STEPS, 1.0 if self.cooldown[1] > 0 else 0.0)
        G = r[ix.game]
        o[27:38] = (G[0] / 3, min(G[1], 3) / 3, min(G[2], 2) / 2, G[3], G[4], G[5], min(G[6], 4) / 4, G[7],
                    min(G[8], 10) / 10, G[9], math.log10(r[ix.score] + 1) / 7)
        o[38:38 + N_MISSION] = 0.0
        o[38 + int(min(max(r[ix.mission], 0), N_MISSION - 1))] = 1.0
        o[38 + N_MISSION:38 + N_MISSION + N_LANES] = r[ix.lanes]
        for i in range(2):
            need = R.MIN_HOLD_STEPS if self.cmd[i] else R.MIN_RELEASE_STEPS
            o[38 + N_MISSION + N_LANES + i] = 1.0 if self.cmd_age[i] < need else 0.0
        np.clip(o, -5, 5, out=o)
        return o

    def begin(self, raw):
        self.raw = np.asarray(raw, dtype=np.float64)
        self._init_flipper_state()
        self.up_flag = True
        self.last_progress = self.raw[self.ix.time]
        self.lane_lit = self._lane_count()
        self.ep_paid_hits = 0
        self.ep_steps = 0
        self.ep_forced = 0
        return self._obs()

    def prepare(self, action):
        cfg = R.stage_config(self.stage)
        reward = 0.0
        pressed = [0, 0]
        for i in range(2):
            cmd = int(action[i])
            need = R.MIN_HOLD_STEPS if self.cmd[i] else R.MIN_RELEASE_STEPS
            if self.cmd_age[i] < need:
                cmd = self.cmd[i]
            if self.cooldown[i] > 0:
                self.cooldown[i] -= 1
                cmd = 0
                self.hold[i] = 0
            elif cmd:
                self.hold[i] += 1
                if self.hold[i] > R.HOLD_MAX_STEPS:
                    cmd = 0
                    self.hold[i] = 0
                    self.cooldown[i] = R.HOLD_COOLDOWN_STEPS
                    reward += cfg.forced_release
                    self.ep_forced += 1
            else:
                self.hold[i] = 0
            if cmd != self.cmd[i]:
                self.cmd[i] = cmd
                self.cmd_age[i] = 0
            else:
                self.cmd_age[i] += 1
            pressed[i] = cmd
        return pressed, reward

    def finish(self, raw, pressed, reward):
        cfg = R.stage_config(self.stage)
        ix = self.ix
        prev = self.raw
        now = self.raw = np.asarray(raw, dtype=np.float64)

        B = now[ix.balls]
        if np.any((B[:, 0] > 0.5) & (B[:, 2] < R.RALLY_Y)):
            self.up_flag = True

        d_act = (now[ix.hits_l_act] - prev[ix.hits_l_act]) + (now[ix.hits_r_act] - prev[ix.hits_r_act])
        d_pas = (now[ix.hits_l_pas] - prev[ix.hits_l_pas]) + (now[ix.hits_r_pas] - prev[ix.hits_r_pas])
        progress = False
        if (d_act > 0 or d_pas > 0) and self.up_flag:
            reward += cfg.hit_active if d_act > 0 else cfg.hit_passive
            self.up_flag = False
            self.ep_paid_hits += 1
            progress = True

        d_score = now[ix.score] - prev[ix.score]
        if cfg.score_unit > 0 and d_score > 0:
            reward += min(d_score / cfg.score_unit, cfg.score_clip)
        if d_score > 0 and self.stage == "score":
            progress = True

        lit = self._lane_count()
        if lit > self.lane_lit:
            reward += cfg.lane_light * (lit - self.lane_lit)
        self.lane_lit = lit

        if now[ix.drains] > prev[ix.drains]:
            reward += cfg.drain
            progress = True
            self.up_flag = True
        if now[ix.launches] > prev[ix.launches]:
            progress = True
            self.up_flag = True

        reward += cfg.press_cost * (pressed[0] + pressed[1])
        if now[ix.n_balls] > 0:
            reward += cfg.alive

        t = now[ix.time]
        if progress:
            self.last_progress = t
        self.ep_steps += 1

        terminated = now[ix.game_over] > 0.5
        truncated = False
        if not terminated:
            if t - self.last_progress > R.STALL_SECONDS:
                truncated = True
                reward += cfg.stall
            elif t > MAX_GAME_SECONDS:
                truncated = True

        info = {}
        if terminated or truncated:
            hits = now[ix.hits_l_act] + now[ix.hits_r_act]
            info["episode_stats"] = {
                "score": float(now[ix.score]), "paid_hits": self.ep_paid_hits, "hits_active": float(hits),
                "drains": float(now[ix.drains]), "game_time": float(t), "forced": self.ep_forced,
                "launches": float(now[ix.launches]), "steps": self.ep_steps, "stalled": bool(truncated),
            }
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for attempt in range(3):
            try:
                if self.game is None:
                    self.start_game()
                raw = self.game.reset()
                break
            except (ConnectionError, OSError, RuntimeError):
                self.game = None
                if attempt == 2:
                    raise
        return self.begin(raw), {}

    def step(self, action):
        pressed, reward = self.prepare(action)
        try:
            raw = self.game.step(pressed[0], pressed[1], self.frame_skip)
        except (ConnectionError, OSError):
            self.game = None
            return np.zeros(OBS_DIM, np.float32), 0.0, False, True, {"crashed": True}
        return self.finish(raw, pressed, reward)

    def close(self):
        if self.game is not None:
            self.game.close()
            self.game = None
