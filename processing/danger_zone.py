import numpy as np


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def to_list(self) -> list[int]:
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

    def contain(self, target: Point) -> bool:
        intersections = self.__count_intersections(target)

        return intersections % 2 == 1

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
        hull_len = len(self.hull)

        for i in range(hull_len):
            if self.__is_intersect_with_ray(target, self.hull[i], self.hull[(i + 1) % hull_len]):
                intersection_counter += 1

        return intersection_counter


def main():
    danger_zone = DangerZone(
        [Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)])

    target = Point(0.9999, 0.00000001)

    print(danger_zone.contain(target))


if __name__ == "__main__":
    main()
