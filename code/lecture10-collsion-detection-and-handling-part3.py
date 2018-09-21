# lecture10-collsion-detection-and-handling-part1.py ---
#
# Filename: lecture10-collsion-detection-and-handling-part1.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Sep 19 17:48:29 2018 (-0700)
# Version:
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#

# Code:

import time

import cv2
import numpy as np

FPS = np.array(60, dtype=np.float)
GRAVITY = 500
# GRAVITY = 12000


class block(object):
    """floor class"""

    def __init__(self, center, width, height, color, acceleration):

        self.center = np.array(center, dtype=np.float)
        self.updated = True
        self.color = color

        # The *state* that we will change
        self.acceleration = np.array(acceleration, dtype=np.float)
        self.velocity = np.array([0, 0], dtype=np.float)

        self.width = width
        self.height = height
        # the points determining this block, with the center at origin
        self.pts = np.array([
            [-self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, +self.height * 0.5],
            [-self.width * 0.5, +self.height * 0.5],
        ])

    def update(self):
        # Update velocity
        self.velocity += (self.acceleration / FPS)
        # Update position
        self.center += (self.velocity / FPS)
        # Record update so that we don't recalculate
        self.updated = True

    def get_points(self):

        if self.updated:
            self.abs_pts = np.round(
                self.center[None] - self.pts).astype(np.int)
            self.updated = False

        return self.abs_pts

    def draw(self, canvas):

        cv2.fillConvexPoly(canvas, self.get_points()[None], self.color)

def process_collision(obj_list):

    # Sort using the python built-in according to start points
    x_sorted = sorted(
        enumerate(obj_list),
        key=lambda idx_obj: idx_obj[1].get_points()[:, 0].min())
    # convert x_sorted into indice arrays
    x_sorted = [_x[0] for _x in x_sorted]

    # Do the same for y_sorted
    y_sorted = sorted(
        enumerate(obj_list),
        key=lambda idx_obj: idx_obj[1].get_points()[:, 1].min())
    y_sorted = [_y[0] for _y in y_sorted]

    # List of things that are overlapping for x
    x_overlap = []
    # Going through the order see if there is overlap
    for i in range(len(x_sorted) - 1):
        cur_end = obj_list[x_sorted[i]].get_points()[:, 0].max()
        next_start = obj_list[x_sorted[i + 1]].get_points()[:, 0].min()
        # If overlap, add to list
        if cur_end > next_start:
            min_idx = min(x_sorted[i], x_sorted[i + 1])
            max_idx = max(x_sorted[i], x_sorted[i + 1])
            x_overlap.append((min_idx, max_idx))

    # Do the same for y
    y_overlap = []
    # Going through the order see if there is overlap
    for i in range(len(y_sorted) - 1):
        cur_end = obj_list[y_sorted[i]].get_points()[:, 1].max()
        next_start = obj_list[y_sorted[i + 1]].get_points()[:, 1].min()
        # If overlap, add to list
        if cur_end > next_start:
            min_idx = min(y_sorted[i], y_sorted[i + 1])
            may_idx = max(y_sorted[i], y_sorted[i + 1])
            y_overlap.append((min_idx, max_idx))

    # Check if there's a common element in x and y overlaps
    collision = []
    for overlap in x_overlap:
        if overlap in y_overlap:
            collision += [overlap]
     
    # Deal with collision
    for pair in collision:
        for idx in pair:
            obj = obj_list[idx]
            # Undo update
            obj.center -= (obj.velocity / FPS)
            obj.velocity -= (obj.acceleration / FPS)

def main():

    # List of objects
    obj_list = [
        block((50, 10), 10, 10, (255, 0, 0), (0, GRAVITY)),
        block((50, 70), 100, 5, (0, 200, 200), (0, 0))
    ]

    while True:

        # For the FPS counter
        time_start = time.time()

        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        for obj in obj_list:
            obj.update()
        process_collision(obj_list)
        for obj in obj_list:
            obj.draw(canvas)

        # Measure compute/rendering time
        time_comp = time.time() - time_start
        time_wait = 1000.0 / FPS - 1000.0 * time_comp
        # print(time_comp, time_wait)
        key = cv2.waitKey(int(time_wait))

        # For the FPS counter
        cur_fps = 1.0 / (time.time() - time_start)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            canvas, str(int(round(cur_fps))), (5, 10), font, 0.3, (0, 255, 0))

        # Final display
        cv2.imshow(
            "canvas", cv2.resize(canvas, (500, 500), interpolation=cv2.INTER_NEAREST))

        # Deal with keyboard input
        if 113 == key:
            print("done")
            break
        elif 114 == key:
            obj_list = [
                block((50, 10), 10, 10, (255, 0, 0), (0, GRAVITY)),
                block((50, 70), 100, 5, (0, 200, 200), (0, 0))
            ]
            print("reset")

    pass


if __name__ == "__main__":
    main()
    exit(0)


#
# lecture10-collsion-detection-and-handling-part1.py ends here
