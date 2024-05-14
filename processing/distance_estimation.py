import time

import cv2
from cv2.typing import MatLike
import numpy as np
import matplotlib.pyplot as plt
import torch
from io import BytesIO


def find_vanishing_point(lines):
    """ Find the vanishing point from a set of lines. """
    if not lines.any():
        return None

    lines = lines.reshape(-1, 4)
    points = []
    for (x1, y1, x2, y2) in lines:
        for (x3, y3, x4, y4) in lines:
            denom = (x1-x2) * (y3-y4) - (y1-y2) * (x3-x4)
            if denom == 0:
                continue  # Parallel or coincident lines
            px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
            py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
            points.append([px, py])

    if not points:
        return None

    # Compute the average point as the vanishing point
    vp = np.mean(points, axis=0)
    return vp


def show_image(image, window_name: str):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def estimate_focal_length(image):
    """ Estimate the camera's focal length based on vanishing points in the image. """
    if image is None:
        raise FileNotFoundError(f"Image not found")

    blured = cv2.GaussianBlur(image, (5, 5), 0)
    show_image(blured, "blured image")

    # Edge detection
    edges = cv2.Canny(blured, 100, 200, apertureSize=3, L2gradient=True)
    show_image(edges, "edged image")

    # Line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=125, minLineLength=125, maxLineGap=10)
    print("Lines shape: {lines.shape}")

    # DEMO
    lined_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lined_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    show_image(lined_image, "lined image")

    if lines is not None:
        vp = find_vanishing_point(lines)
        if vp is not None:
            print(f"vp: {vp}")
            vanishing_point_image = lined_image.copy()
            cv2.circle(vanishing_point_image,
                       (int(vp[0]), int(vp[1])), 10, (0, 0, 255), 5)
            show_image(vanishing_point_image, "vanishing image")

            # Assuming the principal point is at the center
            cx, cy = image.shape[1] / 2, image.shape[0] / 2
            focal_length = np.sqrt(np.abs((vp[0] - cx)**2 + (vp[1] - cy)**2))
            return focal_length
    return None


class MiDaSDistanceEstimator:
    def __init__(self, model_type: str = "DPT_Large", verbose: bool = False):
        self.model_type = model_type
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, verbose=verbose)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", verbose=verbose)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def predict(self, img: MatLike):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        normalized_depth_map = cv2.normalize(
            depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized_depth_map


def create_side_by_side(image, depth, grayscale):
    """
    Take an RGB image and depth map and place them side by side. This includes a proper normalization of the depth map
    for better visibility.

    Args:
        image: the RGB image
        depth: the depth map
        grayscale: use a grayscale colormap?

    Returns:
        the image and depth map place side by side
    """
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(
            np.uint8(right_side), cv2.COLORMAP_PLASMA)

    if image is None:
        return right_side
    else:
        return np.concatenate((image, right_side), axis=1)


def create_3d_scene_from_depth_map(depth_map):
    height, width = depth_map.shape

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    xx = xx.flatten()
    yy = yy.flatten()
    zz = depth_map.flatten()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.gca().invert_yaxis()
    ax.view_init(125, -100, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Depth')

    ax.scatter(xx, yy, zz, c=zz, cmap='plasma')

    plt.savefig('3d_scene.png', dpi=600)
    plt.show()

    plt.close(fig)


def webcam_main():
    model_type = "DPT_Large"
    # model_type = "DPT_Hybrid"
    # model_type = "MiDaS_small"
    distance_estimator = MiDaSDistanceEstimator(model_type)

    with torch.no_grad():
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            if frame is not None:
                prediction = distance_estimator.predict(frame)

                original_image_bgr = None
                content = create_side_by_side(
                    original_image_bgr, prediction, False)

                three_d_image = create_3d_scene_from_depth_map(prediction)

                cv2.imshow("Original", frame)
                cv2.imshow('Distance estimation', content)
                cv2.imshow("3D Depth map", three_d_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # q key
                    break

        video.release()
        cv2.destroyAllWindows()


def main():
    # Example usage
    image_path = "/home/throder/disser/PPE_Detection/examples/demo/arlan.jpg"
    image = cv2.imread(image_path, 0)
    show_image(image, "arlan.jpg")

    focal_length = estimate_focal_length(image)
    print(f"Estimated Focal Length: {focal_length}")


def main2():
    distance_predictor = MiDaSDistanceEstimator()

    image_path = "/home/throder/disser/PPE_Detection/examples/demo/arlan.jpg"
    image = cv2.imread(image_path, 0)

    depth_map = distance_predictor.predict(image)
    print(f"{depth_map.shape=}")
    original_image_bgr = None

    cv2.imshow("Original", image)
    cv2.imshow("Distance estimation", depth_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    create_3d_scene_from_depth_map(depth_map)


if __name__ == "__main__":
    # webcam_main()
    main2()
