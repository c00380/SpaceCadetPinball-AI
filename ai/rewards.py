from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    hit_active: float
    hit_passive: float
    score_unit: float
    score_clip: float
    drain: float
    press_cost: float
    forced_release: float
    stall: float
    alive: float
    lane_light: float


STAGES = {
    "hit": RewardConfig(
        hit_active=1.0, hit_passive=0.1, score_unit=0.0, score_clip=0.0,
        drain=-1.0, press_cost=-0.003, forced_release=-0.05, stall=-1.0, alive=0.001,
        lane_light=0.0),
    "score": RewardConfig(
        hit_active=0.1, hit_passive=0.0, score_unit=5000.0, score_clip=20.0,
        drain=-2.0, press_cost=-0.003, forced_release=-0.05, stall=-2.0, alive=0.001,
        lane_light=0.2),
}

FRAME_SKIP = 6
MIN_HOLD_STEPS = 2
MIN_RELEASE_STEPS = 2

HOLD_MAX_STEPS = 12
HOLD_COOLDOWN_STEPS = 3

STALL_SECONDS = 40.0

RALLY_Y = 3.0


def stage_config(stage):
    return STAGES[stage]
