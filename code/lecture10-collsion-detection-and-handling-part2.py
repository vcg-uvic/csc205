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


class block(object):
    """floor class"""

    def __init__(self, center, width, height, color, acceleration):

        self.center = np.array(center, dtype=np.float)
        self.color = color
        # ------------------------------------------------------------
        # New code from this part
        self.acceleration = np.array(acceleration, dtype=np.float)
        self.velocity = np.array([0, 0], dtype=np.float)
        # ------------------------------------------------------------

        self.width = width
        self.height = height
        # the points determining this block, with the center at origin
        self.pts = np.array([
            [-self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, +self.height * 0.5],
            [-self.width * 0.5, +self.height * 0.5],
        ])

    # ------------------------------------------------------------
    # New code from this part
    def update(self):
        # Update velocity
        self.velocity += (self.acceleration / FPS)
        # Udpate position
        self.center += (self.velocity / FPS)
    # ------------------------------------------------------------

    def get_points(self):

        return np.round(self.center[None] - self.pts).astype(np.int)

    def draw(self, canvas):

        cv2.fillConvexPoly(canvas, self.get_points()[None], self.color)


def main():

    obj_list = []

    # Add objects to list of objects
    obj_list += [
        block((50, 10), 10, 10, (255, 0, 0), (0, 500)),
        block((50, 70), 100, 5, (0, 200, 200), (0, 0))
    ]

    while True:

        # For the FPS counter
        time_start = time.time()

        canvas = np.ones((100, 100, 3), dtype=np.uint8) * 255
        for obj in obj_list:
            obj.update()
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
    pass


if __name__ == "__main__":
    main()
    exit(0)


#
# lecture10-collsion-detection-and-handling-part1.py ends here
