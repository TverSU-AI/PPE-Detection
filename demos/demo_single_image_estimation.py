# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu,Hao-Shu Fang
# -----------------------------------------------------

"""Script for single-image demo."""
import argparse
import os
import json
import sys
import importlib
from pathlib import Path
from typing import Callable
from cv2.typing import MatLike

import cv2
import torch


def import_parents(level=1):
    global __package__
    file = Path(__file__).resolve()
    parent, top = file.parent, file.parents[level]

    sys.path.append(str(top))
    try:
        sys.path.remove(str(parent))
    except ValueError:
        pass

    __package__ = '.'.join(parent.parts[len(top.parts):])
    importlib.import_module(__package__)


if __name__ == '__main__' and __package__ is None:
    import_parents()

    from processing.drawing import draw_zone
    from processing.danger_zone import DangerZone, Point
    from processing.alpha_pose_pose_estimation import PoseEstimator


def parse_arguments():
    """----------------------------- Demo options -----------------------------"""
    parser = argparse.ArgumentParser(description='AlphaPose Single-Image Demo')
    parser.add_argument('--cfg', type=str, required=False,
                        help='experiment configure file name', default="../AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_2x.yaml")
    parser.add_argument('--checkpoint', type=str, required=False,
                        help='checkpoint file name', default="../AlphaPose/pretrained_models/halpe26_fast_res50_256x192.pth")
    parser.add_argument('--detector', dest='detector',
                        help='detector name', default="yolo")
    parser.add_argument('--image', dest='inputimg',
                        help='image-name', default="/home/throder/disser/PPE_Detection/examples/demo/1.jpg")
    parser.add_argument('--save_img', default=True, action='store_true',
                        help='save result as image')
    parser.add_argument('--vis', default=False, action='store_true',
                        help='visualize image')
    parser.add_argument('--showbox', default=False, action='store_true',
                        help='visualize human bbox')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='add speed profiling at screen output')
    parser.add_argument('--format', type=str,
                        help='save in the format of cmu or coco or openpose, option: coco/cmu/open', default="coco")
    parser.add_argument('--min_box_area', type=int, default=0,
                        help='min box area to filter out')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                        help='save the result json as coco format, using image index(int) instead of image name(str)')
    parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                        help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='enable flip testing')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='print detail information')
    parser.add_argument('--vis_fast', dest='vis_fast',
                        help='use fast rendering', action='store_true', default=True)
    """----------------------------- Tracking options -----------------------------"""
    parser.add_argument('--pose_flow', dest='pose_flow',
                        help='track humans in video with PoseFlow', action='store_true', default=False)
    parser.add_argument('--pose_track', dest='pose_track',
                        help='track humans in video with reid', action='store_true', default=False)

    args = parser.parse_args()

    args.gpus = [int(args.gpus[0])] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device(
        "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

    args.model_cfg = "../AlphaPose/detector/yolo/cfg/yolov3-spp.cfg"
    args.model_weights = "../AlphaPose/detector/yolo/data/yolov3-spp.weights"

    return args


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["drawing"] = True
        print("PRESSED AT", x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if param["drawing"] == True:
            print("MOVING")

    elif event == cv2.EVENT_LBUTTONUP:
        param["drawing"] = False
        print("UNPRESSED AT", x, y)


def set_mouse_callback(callback: Callable, window_name: str = "image", param: dict | None = None):
    cv2.setMouseCallback(window_name, callback, param)


def use_webcam(pose_estimator: PoseEstimator, zone: DangerZone, *, use_mouse_callback: bool = True):
    vid = cv2.VideoCapture(0)
    image: MatLike | None = None
    param = {
        "drawing": False
    }

    if use_mouse_callback:
        cv2.namedWindow('webcam')
        set_mouse_callback(onMouse, "webcam", param)

    while (True):
        ret, frame = vid.read()

        packed_json, posed_frame = pose_estimator.process(frame)

        # image = draw_zone(posed_frame, zone)
        image = posed_frame

        cv2.imshow('webcam', image)

        for people in packed_json["persons"]:
            name = people["person_id"]
            keypoints = people["keypoints"]

            if zone.contain_pose(keypoints):
                print(f"People[{name}] inside danger zone!")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def example():
    outputpath = "examples/res/"
    if not os.path.exists(outputpath + '/vis'):
        os.mkdir(outputpath + '/vis')

    args = parse_arguments()

    image_name = args.inputimg

    pose_estimator = PoseEstimator(args)

    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    packed_json, posed_image = pose_estimator.process(image, image_name)

    cv2.imwrite(os.path.join(outputpath, 'vis',
                os.path.basename(image_name)[:-4] + "_posed.jpg"), posed_image)

    # zone = DangerZone(
    #     hull=[
    #         Point(320, 350),
    #         Point(305, 400),
    #         Point(405, 400),
    #         Point(390, 350)
    #     ],
    #     upper_shift=Point(0, -300)
    # )

    zone = DangerZone(
        hull=[
            Point(470, 360),
            Point(415, 485),
            Point(630, 530),
            Point(670, 395)
        ],
        upper_shift=Point(0, -350)
    )

    image_with_zone = draw_zone(posed_image, zone)
    cv2.imwrite(os.path.join(outputpath, 'vis',
                os.path.basename(image_name)[:-4] + "_zoned.jpg"), image_with_zone)

    with open(os.path.join(outputpath, os.path.basename(image_name)[:-4] + "_result_packed.json"), 'w') as json_file:
        json_file.write(json.dumps(packed_json, indent=2))

    for people in packed_json["persons"]:
        name = people["person_id"]
        keypoints = people["keypoints"]

        if zone.contain_pose(keypoints):
            print(f"People[{name}] inside danger zone!")

    cv2.imshow("frame", image_with_zone)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # use_webcam(pose_estimator, zone)


if __name__ == "__main__":
    example()
