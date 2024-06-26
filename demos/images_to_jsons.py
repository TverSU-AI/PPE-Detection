import sys
import os
import importlib
import argparse
import json
from pathlib import Path

import cv2
from cv2.typing import MatLike


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


if __name__ == '__main__':  # and __package__ is None:
    import_parents()

    from processing.packed_json_generator import JSONGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='image_to_jsons demo arg parser')
    parser.add_argument("--image_dir", type=str, required=False,
                        help="path to images dir", default=str(Path("examples", "demo")))

    parser.add_argument("--output_dir", type=str, required=False,
                        help="path to jsons output dir", default=str(Path("examples", "res")))

    args = parser.parse_args()

    return args


def show_posed_image(posed_image: MatLike, name: str) -> None:
    cv2.imshow(name, posed_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def save_packed_json(packed_json: dict, output_path: str) -> None:
    filepath = str(Path(output_path, 'jsons',
                   f"{packed_json['image_id']}.json"))
    with open(filepath, "w") as f:
        json.dump(packed_json, f, indent=2)


def save_posed_image(posed_image: MatLike, output_path: str, image_name: str) -> None:
    cv2.imwrite(os.path.join(output_path, 'posed_images',
                os.path.basename(image_name)), posed_image)


def create_output_dirs(output_path: str) -> None:
    paths = [
        output_path,
        f"{output_path}/jsons",
        f"{output_path}/posed_images"
    ]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


def main() -> None:

    args = parse_arguments()

    path_to_images_dir = args.image_dir
    output_path = args.output_dir

    create_output_dirs(output_path)

    json_generator = JSONGenerator()

    for packed_json, posed_image in json_generator(path_to_images_dir):
        show_posed_image(posed_image, packed_json["image_id"])
        save_posed_image(posed_image, output_path, packed_json["image_id"])
        save_packed_json(packed_json, output_path)


if __name__ == "__main__":
    main()
