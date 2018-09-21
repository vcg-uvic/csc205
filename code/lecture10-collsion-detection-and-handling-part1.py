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

FPS = 60

class block(object):
    """floor class"""

    def __init__(self, center, width, height, color):

        self.center = np.array(center)
        self.color = color

        self.width = width
        self.height = height
        # the points determining this block, with the center at origin
        self.pts = np.array([
            [-self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, +self.height * 0.5],
            [-self.width * 0.5, +self.height * 0.5],
        ])


    def get_points(self):
        
        return np.round(self.center[None] - self.pts).astype(np.int)

    def draw(self, canvas):

        cv2.fillConvexPoly(canvas, self.get_points()[None], self.color)


def main():

    obj_list = []

    # Add objects to list of objects
    obj_list += [
        block((50,50), 10, 10, (255, 0, 0)),
        block((50,70), 100, 5, (0, 200, 200))
    ]

    while True:

        canvas = np.ones((100,100,3), dtype=np.uint8) * 255
        for obj in obj_list:
            obj.draw(canvas)

        cv2.imshow(
            "canvas", cv2.resize(
                canvas, (500, 500), interpolation=cv2.INTER_NEAREST))

        key = cv2.waitKey(round(1000/FPS))
        if 113 == key:
            print("done")
            break

    pass



if __name__ == "__main__":
    main()
    exit(0)


#
# lecture10-collsion-detection-and-handling-part1.py ends here
