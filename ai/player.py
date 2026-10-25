import os
import sys

import numpy as np
from stable_baselines3 import PPO

import model_store as store
from pinball_env import PinballEnv


def play(args):
    path = args.model or store.MODEL_PATH
    if not os.path.exists(path):
        sys.exit(f"model not found: {path}\nTrain first: python ai/pinball_ai.py train")
    model = PPO.load(path, device="cpu")

    env = PinballEnv(stage=store.load_meta()["stage"], headless=False)
    scores = []
    try:
        while True:
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info.get("crashed"):
                    print("game window closed")
                    return
            stats = info["episode_stats"]
            scores.append(stats["score"])
            print(f"game {len(scores)}: score {stats['score']:,.0f}  time {stats['game_time']:.0f}s  "
                  f"avg {np.mean(scores):,.0f}  max {max(scores):,.0f}", flush=True)
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        env.close()
