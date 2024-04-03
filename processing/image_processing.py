import json
import base64
from typing import Any

import cv2
import numpy as np
from cv2.typing import MatLike


def image_encode(image: MatLike) -> str:
    if image is None:
        return None

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    save_format = ".png"

    _, buffer = cv2.imencode(save_format, image, encode_params)
    data = np.array(buffer)
    encoded_image = base64.b64encode(data)
    text_encoded_image = encoded_image.decode("ascii")

    return text_encoded_image


def image_decode(img_base64: str) -> MatLike:
    img_as_np = np.frombuffer(base64.b64decode(img_base64), np.uint8)
    image = cv2.imdecode(img_as_np, cv2.IMREAD_COLOR)

    return image


def pack_json(image: str | MatLike, alphapose_results: dict) -> dict[str, Any]:
    if type(image) is str:
        encoded_image = image
    else:
        encoded_image = image_encode(image)

    if type(alphapose_results[0]) is str:
        image_id = alphapose_results[0]
    elif "image_id" in alphapose_results[0].keys():
        image_id = alphapose_results[0]["image_id"]
    else:
        image_id = "default_id"

    main_json: dict[str, Any] = {
        "persons": [],
        "image_id": image_id,
        "encoded_image": f"{encoded_image}",
        "format": "coco"
    }

    if type(alphapose_results[0]) is str:
        return main_json

    for i, person_json in enumerate(alphapose_results):
        json_obj = {
            "person_id": f"{i}",
            "keypoints": person_json["keypoints"],
            "bounding_box": person_json["box"]
        }

        main_json["persons"].append(json_obj)

    return main_json


def save_json_to_file(filepath: str, json_obj: dict[str, Any]) -> None:
    with open(filepath, "w") as f_out:
        json.dump(json_obj, f_out)


def load_json_from_file(filepath: str) -> dict[str, Any]:
    with open(filepath, "r") as f_in:
        dictionary = json.load(f_in)

    return dictionary


def __is_similar(image1: MatLike, image2: MatLike) -> bool:
    return image1.shape == image2.shape and not np.bitwise_xor(image1, image2).any()


def __usage_example() -> None:
    prefix_path = "/home/throder/disser/AlphaPose/processing"
    json_path = f"{prefix_path}/obj.json"
    # f"{prefix_path}/1.jpg"
    image_path = "/home/throder/Загрузки/putin-t-pose.jpg"
    image = cv2.imread(image_path)

    encode_decode = image_decode(image_encode(image))

    if not __is_similar(image, encode_decode):
        print("Encode-decoded image differs from original!")
        print(len(image), len(encode_decode))
        return

    encoded_image = image_encode(image)

    alphapose_results = load_json_from_file(
        f"{prefix_path}/../examples/res/alphapose-results.json")

    json_obj = pack_json(encoded_image, alphapose_results)
    save_json_to_file(json_path, json_obj)

    json_obj = load_json_from_file(json_path)
    readed_encoded_image = json_obj["encoded_image"]

    if readed_encoded_image != encoded_image:
        print("Readed encoded image differs from original!")
        print(len(encoded_image), len(readed_encoded_image))
        return

    decoded_image = image_decode(readed_encoded_image)

    if not __is_similar(image, decoded_image):
        print("Decoded image differs from original!")
        print(len(image), len(decoded_image))
        return


if __name__ == "__main__":
    __usage_example()
