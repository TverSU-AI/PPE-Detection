import cv2
from cv2.typing import MatLike
import numpy as np

from processing.danger_zone import DangerZone, Point


def draw_hull(image: MatLike, zone: DangerZone, color: tuple[int, int, int] = (0, 0, 255)) -> MatLike:
    img = image.copy()
    hull = zone.to_points_list()

    shapes = np.zeros_like(img, np.uint8)
    cv2.fillPoly(shapes, [hull], color)

    alpha = -1
    mask = shapes.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, shapes, 1 - alpha, 0)[mask]

    for point in zone.hull:
        cv2.circle(img, point.to_list(), 5, color, cv2.FILLED)

    cv2.polylines(img, [hull], True, color, 2)

    return img


def draw_convex_hull(image: MatLike, zone: DangerZone, color: tuple[int, int, int] = (0, 0, 255)) -> MatLike:
    img = image.copy()

    hull = np.array([point.to_list() for point in zone.convex_hull])

    cv2.polylines(img, [hull], True, color, 2)

    return img


def draw_reference_lines(image: MatLike, zone: DangerZone, color: tuple[int, int, int] = (0, 0, 255)) -> MatLike:
    img = image.copy()

    edges = [[[p.x, 0], p.to_list()] for p in zone.hull]

    for edge in edges:
        cv2.line(img, edge[0], edge[1], color, 1)

    return img


def draw_zone(image: MatLike, zone: DangerZone) -> MatLike:
    rgba_canvas = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

    image_with_hull = draw_hull(rgba_canvas, zone)

    zone_height = zone.upper_shift
    upper_zone = zone.shift(zone_height) \
                     .shift(zone_height // Point(1, -10), [True, False, False, True])
    image_with_hull = draw_hull(image_with_hull, upper_zone)

    image_reference_lines = draw_reference_lines(image_with_hull, zone)
    return image_reference_lines
