import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    t = sub.add_parser("train", help="train with parallel headless games")
    t.add_argument("--envs", type=int, default=32,
                   help="parallel games (they mostly wait, so more than CPU cores is fine)")
    t.add_argument("--workers", type=int, default=4, help="Python processes that drive the games (1 = in-process)")
    t.add_argument("--threads", type=int, default=max(2, (os.cpu_count() or 4) // 2), help="PyTorch threads")
    t.add_argument("--steps", type=int, default=20_000_000, help="timesteps to train in this run (Ctrl+C to stop)")
    t.add_argument("--stage", choices=["auto", "hit", "score"], default="auto",
                   help="auto: learn to hit the ball first, then maximise the score")
    t.add_argument("--hit-max-steps", type=int, default=1_500_000, help="most steps spent in stage hit")
    t.add_argument("--n-steps", type=int, default=None, help="rollout length per game (default 4096 / envs)")
    t.add_argument("--snapshot-every", type=int, default=500_000, help="steps between snapshots")
    t.add_argument("--fresh", action="store_true", help="delete the saved model and start over")
    t.add_argument("--no-bc", action="store_true", help="skip pre-training on the heuristic (behaviour cloning)")
    t.add_argument("--bc-steps", type=int, default=60_000, help="demonstration steps for behaviour cloning")

    p = sub.add_parser("play", help="watch the model play in the game window")
    p.add_argument("--model", help="model zip (default model/model.zip; e.g. a file from model/snapshots)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of using the best one")
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        from trainer import train
        train(args)
    else:
        from player import play
        play(args)


if __name__ == "__main__":
    main()
