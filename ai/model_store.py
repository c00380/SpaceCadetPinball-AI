import glob
import json
import os

from game_client import ROOT

MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.zip")
META_PATH = os.path.join(MODEL_DIR, "meta.json")
LOG_PATH = os.path.join(MODEL_DIR, "train_log.csv")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
SNAP_DIR = os.path.join(MODEL_DIR, "snapshots")
BEST_PATH = os.path.join(SNAP_DIR, "best_score.zip")

SNAPSHOT_KEEP = 8
SNAPSHOT_MAX_MB = 64


def new_meta():
    return {"stage": "hit", "timesteps": 0, "best_score": 0.0, "history": []}


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            return json.load(f)
    return new_meta()


def save_meta(meta):
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = META_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, META_PATH)


def save_snapshot(model, steps, stage):
    os.makedirs(SNAP_DIR, exist_ok=True)
    model.save(os.path.join(SNAP_DIR, f"snap_{steps:010d}_{stage}.zip"))
    snapshots = sorted(glob.glob(os.path.join(SNAP_DIR, "snap_*.zip")))
    for path in snapshots[:-SNAPSHOT_KEEP]:
        os.remove(path)
    snapshots = snapshots[-SNAPSHOT_KEEP:]
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(SNAP_DIR, "*.zip")))
    while snapshots and total > SNAPSHOT_MAX_MB * 1024 * 1024:
        total -= os.path.getsize(snapshots[0])
        os.remove(snapshots.pop(0))


def clear():
    removed = 0
    for pattern in ("model.zip", "meta.json", "train_log.csv", "vecnormalize.pkl", os.path.join("snapshots", "*.zip")):
        for path in glob.glob(os.path.join(MODEL_DIR, pattern)):
            os.remove(path)
            removed += 1
    print(f"[train] --fresh: removed {removed} old file(s) from {MODEL_DIR}")
