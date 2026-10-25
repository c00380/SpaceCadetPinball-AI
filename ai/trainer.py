import csv
import os
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

import model_store as store
from fast_policy import install_fast_inference
from heuristic import heuristic_action
from pinball_env import PinballEnv
from pinball_vec import PinballParallelVecEnv, PinballVecEnv

HIT_TARGET_PER_BALL = 3.0
HIT_WINDOW = 30
SCORE_WINDOW = 100
UPDATE_SAMPLES = 4096
ENT_COEF_START = 0.02
ENT_COEF_END = 0.002
ENT_DECAY_STEPS = 3_000_000


def learning_rate(progress_remaining):
    return 3e-4 * (0.15 + 0.85 * progress_remaining)


def entropy_coef(total_steps):
    fraction = min(1.0, total_steps / ENT_DECAY_STEPS)
    return ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * fraction


def behaviour_clone(model, stage, steps, seed=0):
    rng = np.random.default_rng(seed)
    env = PinballEnv(stage=stage, headless=True)
    obs, _ = env.reset()
    observations, actions = [], []
    while len(observations) < steps:
        action = np.array(heuristic_action(env.raw, env.I))
        observations.append(obs)
        actions.append(action)
        if rng.random() < 0.05:
            action = rng.integers(0, 2, size=2)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    x = torch.as_tensor(np.array(observations), dtype=torch.float32)
    y = torch.as_tensor(np.array(actions), dtype=torch.long)
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    epochs = 8
    for epoch in range(epochs):
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 512):
            batch = order[i:i + 512]
            latent, _ = policy.mlp_extractor(policy.extract_features(x[batch]))
            logits = policy.action_net(latent).reshape(-1, 2)
            loss = loss_fn(logits, y[batch].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"[bc] epoch {epoch + 1}/{epochs} loss {total / len(x):.4f}")


class TrainingMonitor(BaseCallback):

    def __init__(self, args, venv, meta, stage, auto_stage):
        super().__init__()
        self.args, self.venv, self.meta = args, venv, meta
        self.stage, self.auto_stage = stage, auto_stage
        self.episodes = deque(maxlen=SCORE_WINDOW)
        self.episode_count = 0
        self.start_steps = meta["timesteps"]
        self.last_snapshot = meta["timesteps"]
        self.last_print = 0
        self.best_mean = meta["best_score"]
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            stats = info.get("episode_stats")
            if stats:
                self.episodes.append(stats)
                self.episode_count += 1
        return True

    def _mean(self, key, per_ball=False, last=SCORE_WINDOW):
        episodes = list(self.episodes)[-last:]
        if not episodes:
            return 0.0
        if per_ball:
            return float(np.mean([e[key] / max(e["launches"], 1.0) for e in episodes]))
        return float(np.mean([e[key] for e in episodes]))

    def _on_rollout_end(self):
        total = self.start_steps + self.num_timesteps
        self.meta["timesteps"], self.meta["stage"] = total, self.stage
        mean_score = self._mean("score")
        hits_per_ball = self._mean("paid_hits", per_ball=True)
        logged = self.model.logger.name_to_value

        self.model.ent_coef = entropy_coef(total)
        row = dict(
            steps=total, stage=self.stage, episodes=self.episode_count, mean_score=round(mean_score),
            paid_hits_per_ball=round(hits_per_ball, 3), mean_game_time=round(self._mean("game_time"), 1),
            forced_releases=round(self._mean("forced"), 1),
            expl_var=round(float(logged.get("train/explained_variance", 0.0)), 3),
            entropy=round(float(-logged.get("train/entropy_loss", 0.0)), 3),
            approx_kl=round(float(logged.get("train/approx_kl", 0.0)), 4),
            fps=int(self.num_timesteps / (time.time() - self.t0)), ent_coef=round(self.model.ent_coef, 4))
        self._write_log(row)
        if total - self.last_print >= 50_000 or self.last_print == 0:
            self.last_print = total
            print(f"[{self.stage}] steps {total:>10,}  eps {self.episode_count:>5}  score {mean_score:>9,.0f}  "
                  f"rally-hits/ball {hits_per_ball:5.2f}  ev {row['expl_var']:.2f}  ent {row['entropy']:.2f} "
                  f"(coef {row['ent_coef']})  forced {row['forced_releases']:5.1f}  {row['fps']} sps", flush=True)

        self._maybe_switch_stage(total)
        if self.stage == "score" and len(self.episodes) >= SCORE_WINDOW and mean_score > self.best_mean:
            self.best_mean = self.meta["best_score"] = mean_score
            self.model.save(store.BEST_PATH)
        if total - self.last_snapshot >= self.args.snapshot_every:
            self.last_snapshot = total
            store.save_snapshot(self.model, total, self.stage)
        self.save()

    def _maybe_switch_stage(self, total):
        if not (self.auto_stage and self.stage == "hit" and len(self.episodes) >= HIT_WINDOW):
            return
        recent = self._mean("paid_hits", per_ball=True, last=HIT_WINDOW)
        if recent < HIT_TARGET_PER_BALL and total - self.start_steps < self.args.hit_max_steps:
            return
        self.stage = "score"
        self.episodes.clear()
        self.best_mean = self.meta["best_score"] = 0.0
        self.meta["history"].append({"event": "stage hit -> score", "steps": total, "rally_hits_per_ball": recent})
        self.venv.env_method("set_stage", "score")
        print(f"[train] >>> stage 'hit' finished ({recent:.2f} hits/ball): switching to 'score'", flush=True)

    def _write_log(self, row):
        new_file = not os.path.exists(store.LOG_PATH)
        with open(store.LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def save(self):
        self.model.save(store.MODEL_PATH)
        self.venv.save(store.VECNORM_PATH)
        self.meta["stage"] = self.stage
        store.save_meta(self.meta)


def make_vec_env(args, stage):
    if args.workers > 1:
        return PinballParallelVecEnv(args.envs, stage, args.workers)
    return PinballVecEnv(args.envs, stage)


def train(args):
    torch.set_num_threads(args.threads)
    n_steps = args.n_steps or max(64, UPDATE_SAMPLES // args.envs)
    os.makedirs(store.SNAP_DIR, exist_ok=True)

    if args.fresh:
        store.clear()
    meta = store.new_meta() if args.fresh else store.load_meta()
    stage = meta["stage"] if args.stage == "auto" else args.stage

    venv = make_vec_env(args, stage)
    resume = os.path.exists(store.MODEL_PATH) and not args.fresh
    if resume and os.path.exists(store.VECNORM_PATH):
        venv = VecNormalize.load(store.VECNORM_PATH, venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)

    if resume:
        print(f"[train] resuming from {store.MODEL_PATH} (stage={stage}, steps={meta['timesteps']})")
        model = PPO.load(store.MODEL_PATH, env=venv, device="cpu", custom_objects={
            "learning_rate": learning_rate, "lr_schedule": learning_rate, "target_kl": 0.03, "n_steps": n_steps})
        model.ent_coef = entropy_coef(meta["timesteps"])
    else:
        print("[train] new model")
        model = PPO(
            "MlpPolicy", venv, device="cpu", verbose=0, n_steps=n_steps, batch_size=1024, n_epochs=6,
            learning_rate=learning_rate, target_kl=0.03, gamma=0.999, gae_lambda=0.95, clip_range=0.2,
            ent_coef=entropy_coef(0), vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
        if not args.no_bc:
            behaviour_clone(model, stage, args.bc_steps)

    install_fast_inference(model)
    monitor = TrainingMonitor(args, venv, meta, stage, auto_stage=args.stage == "auto")
    try:
        model.learn(total_timesteps=args.steps, callback=monitor)
    except KeyboardInterrupt:
        print("[train] interrupted - saving")
    finally:
        monitor.save()
        venv.close()
    print(f"[train] saved {store.MODEL_PATH}")
