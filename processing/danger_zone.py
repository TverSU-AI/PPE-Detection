import numpy as np


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def to_list(self) -> list[float]:
        return [self.x, self.y]

    def __add__(a, b):
        return Point(a.x + b.x, a.y + b.y)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __floordiv__(self, other):
        return Point(self.x // other.x, self.y // other.y)


class DangerZone:
    def __init__(self, hull: list[Point], upper_shift: Point):
        self.hull = hull
        self.upper_shift = upper_shift

        self.front_hull: list[int] = list()
        self.back_hull: list[int] = list()

        left_ind = np.argmin([point.x for point in self.hull])
        right_ind = np.argmax([point.x for point in self.hull])

        if left_ind < right_ind:
            self.front_hull = list(range(left_ind, right_ind + 1))
        else:
            self.front_hull = list(range(left_ind + 1)) + \
                list(range(right_ind, len(self.hull)))

        self.convex_hull: list[Point] = [self.hull[ind] for ind in self.front_hull] + [
            Point(self.hull[right_ind].x, 0), Point(self.hull[left_ind].x, 0)]

    def contain_point(self, target: Point) -> bool:
        intersections = self.__count_intersections(target)

        return intersections % 2 == 1

    def contain_pose(self, keypoints: list[float], threshold: float = 0.25) -> bool:
        for i in range(0, len(keypoints), 3):
            x, y, c = keypoints[i], keypoints[i + 1], keypoints[i + 2]

            if c <= threshold:
                continue

            if self.contain_point(Point(x, y)):
                return True

        return False

    def to_points_list(self) -> np.ndarray:
        return np.array([point.to_list() for point in self.hull], np.int32)

    def shift(self, p: Point, mask: list[bool] | None = None):
        new_hull: list[Point]

        if mask is None:
            new_hull = [point + p for point in self.hull]
        else:
            new_hull = [
                point + p if p_status else point
                for point, p_status in zip(self.hull, mask)
            ]

        new_zone = DangerZone(
            hull=new_hull,
            upper_shift=self.upper_shift
        )
        return new_zone

    def __is_intersect_with_ray(self, target: Point, a: Point, b: Point) -> bool:
        px = target.x
        py = target.y
        ax = a.x
        ay = a.y
        bx = b.x
        by = b.y

        if by == ay:
            return False

        t: float = (py - ay) / (by - ay)

        if t < 0 or t > 1:
            return False

        if py == min(ay, by):
            return False

        x = (1 - t) * ax + t * bx

        return x >= px

    def __count_intersections(self, target: Point):
        intersection_counter = 0
        hull_len = len(self.convex_hull)

        for i in range(hull_len):
            if self.__is_intersect_with_ray(target, self.convex_hull[i], self.convex_hull[(i + 1) % hull_len]):
                intersection_counter += 1

        return intersection_counter


def main():
    danger_zone = DangerZone(
        [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)])

    target = Point(0.9999, 0.00000001)

    print(danger_zone.contain(target))


if __name__ == "__main__":
    main()
