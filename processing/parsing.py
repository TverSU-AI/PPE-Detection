import json
from types import SimpleNamespace

import torch


def parse_pose_estimator_args(path: str):
    with open(path, "r") as f:
        args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device(
        "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

    return args


if __name__ == "__main__":
    path = "configs/pose_estimator_config.json"
    args = parse_pose_estimator_args(path)

    print(f"{args.cfg=}",
          f"{args.checkpoint=}",
          f"{args.eval=}",
          f"{args.detector=}",
          f"{args.format=}",
          sep="\n")
