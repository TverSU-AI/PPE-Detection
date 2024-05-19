from pathlib import Path

import cv2


from processing.parsing import parse_pose_estimator_args
from processing.alpha_pose_pose_estimation import PoseEstimator


class ImageReader:
    def __init__(self, path_to_images_dir: str):
        def path_to_img(path: str) -> bool:
            format = path.split(".")[-1]

            return format in ["jpg", "jpeg", "png"]

        self.images_paths = [
            str(x) for x in Path(path_to_images_dir).iterdir()
            if path_to_img(str(x))
        ]

    def __iter__(self):
        for image_path in self.images_paths:
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            name = image_path.split("/")[-1]

            yield name, img


class JSONGenerator:
    def __init__(self, *, args_path: str | None = None):
        if args_path is None:
            args_path = str(Path("configs", "pose_estimator_config.json"))
        args = parse_pose_estimator_args(args_path)
        self.pose_estimator = PoseEstimator(args)

    def __iter__(self):
        for name, image in self.images:
            yield self.pose_estimator.process(image, name)

    def __call__(self, path_to_images_dir: str):
        self.images = ImageReader(path_to_images_dir)
        return self
