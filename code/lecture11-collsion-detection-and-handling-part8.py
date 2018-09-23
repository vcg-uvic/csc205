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
CR = 0.1
JUMP_SPEED = 200


class fire(object):
    """floor class"""

    def __init__(self, center, width, height,
                 color, mass, obj_type,
                 acceleration=(0, 0),
                 velocity=(0, 0)):

        self.center = np.array(center, dtype=np.float)
        self.updated = True
        self.color = color
        self.obj_type = obj_type
        self.health = 3

        # The *state* that we will change
        self.acceleration = np.array(acceleration, dtype=np.float)
        self.velocity = np.array(velocity, dtype=np.float)
        self.mass = mass

        self.width = width
        self.height = height
        # the points determining this block, with the center at origin
        self.pts = np.array([
            [-self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, -self.height * 0.5],
            [+self.width * 0.5, +self.height * 0.5],
            [-self.width * 0.5, +self.height * 0.5],
        ])

        self.sprite_sheet = cv2.imread(
            "../images/fire.png", cv2.IMREAD_UNCHANGED)
        self.sprite_idx = 0

    def update(self):
        # Update position
        self.center += (self.velocity / FPS)
        # Update velocity
        self.velocity += (self.acceleration / FPS)
        # Record update so that we don't recalculate
        self.updated = True

    def get_points(self):

        if self.updated:
            scaler = 1.0
            if self.obj_type == "player":
                # Keep narrowing for the whole jump
                scaler = np.exp(-0.01 *
                                np.array([max(0, self.velocity[1]), 0]))
            self.abs_pts = np.round(
                self.center[None] + self.pts * scaler).astype(np.int)
            self.updated = False

        return self.abs_pts

    def draw(self, canvas):

        # convert sprite_idx to y,x
        x = (self.sprite_idx % 8) * 64
        y = (self.sprite_idx // 8) * 128
        cur_sprite = self.sprite_sheet[y:y+128, x:x+64]

        # Resize according to current points
        cur_rect = np.round(self.get_points()).astype(np.int)
        left = cur_rect[0, 0]
        top = cur_rect[0, 1]
        right = cur_rect[1, 0]
        bottom = cur_rect[2, 1]
        cur_sprite = cv2.resize(
            cur_sprite, (right - left + 1, bottom - top + 1))
        # Use the alpha channel and merge with current canvas
        canvas[top:bottom + 1, left:right + 1] = (
            cur_sprite[:, :, :3] * cur_sprite[:, :, 3, None]
            + canvas[top:bottom + 1, left:right + 1] *
            (1 - cur_sprite[:, :, 3, None])
        )

        # Move onto next sprite
        self.sprite_idx += 1
        if self.sprite_idx >= 8 * 4:
            self.sprite_idx = 0



class block(object):
    """floor class"""

    def __init__(self, center, width, height,
                 color, mass, obj_type,
                 acceleration=(0, 0),
                 velocity=(0, 0)):

        self.center = np.array(center, dtype=np.float)
        self.updated = True
        self.color = color
        self.obj_type = obj_type
        self.health = 3

        # The *state* that we will change
        self.acceleration = np.array(acceleration, dtype=np.float)
        self.velocity = np.array(velocity, dtype=np.float)
        self.mass = mass

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
        # Update position
        self.center += (self.velocity / FPS)
        # Update velocity
        self.velocity += (self.acceleration / FPS)
        # Record update so that we don't recalculate
        self.updated = True

    def get_points(self):

        if self.updated:
            scaler = 1.0
            if self.obj_type == "player":
                # Keep narrowing for the whole jump
                scaler = np.exp(-0.01 *
                                np.array([max(0, self.velocity[1]), 0]))
            self.abs_pts = np.round(
                self.center[None] - self.pts * scaler).astype(np.int)
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
        if cur_end >= next_start:
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
        if cur_end >= next_start:
            min_idx = min(y_sorted[i], y_sorted[i + 1])
            max_idx = max(y_sorted[i], y_sorted[i + 1])
            y_overlap.append((min_idx, max_idx))

    # Check if there's a common element in x and y overlaps
    collision = []
    for overlap in x_overlap:
        if overlap in y_overlap:
            collision += [overlap]

    # Deal with collision
    for pair in collision:
        # We will first undo our updates
        for idx in pair:
            obj = obj_list[idx]
            obj.velocity -= (obj.acceleration / FPS)
            obj.center -= (obj.velocity / FPS)

        # Equation from https://en.wikipedia.org/wiki/Coefficient_of_restitution
        va_i = obj_list[pair[0]].velocity
        ma = obj_list[pair[0]].mass
        vb_i = obj_list[pair[1]].velocity
        mb = obj_list[pair[1]].mass

        # We'll use negative number for mass to denote infinite :-) Note that
        # this processing here will cause weird things to happen!
        if ma > 0 and mb > 0:
            va_f = (ma * va_i + mb * vb_i + mb *
                    CR * (vb_i - va_i)) / (ma + mb)
            vb_f = (ma * va_i + mb * vb_i + ma *
                    CR * (va_i - vb_i)) / (ma + mb)
        elif ma > 0:
            va_f = -CR * va_i
            vb_f = vb_i
        elif mb > 0:
            va_f = va_i
            vb_f = -CR * vb_i
        else:
            va_f = va_i
            vb_f = vb_i

        obj_list[pair[0]].velocity = va_f
        obj_list[pair[1]].velocity = vb_f

        # Don't update position, simply apply acceleration
        for idx in pair:
            obj = obj_list[idx]
            obj.velocity += (obj.acceleration / FPS)

        # We also check if the collision was between "player" and "obstacle",
        # and if so, reduce player's health, as well as destroy obstacle.
        player_idx = -1
        obstacle_idx = -1
        for idx in pair:
            obj = obj_list[idx]
            if obj.obj_type == "player":
                player_idx = idx
            elif obj.obj_type == "obstacle":
                obstacle_idx = idx
        if player_idx >= 0 and obstacle_idx >= 0:
            print("Ooow!")
            obj_list[player_idx].health -= 1
            obj_list[obstacle_idx].health = 0


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    obj_list = param["obj_list"]
    if event == cv2.EVENT_LBUTTONUP:
        for obj in obj_list:
            # If player
            if obj.obj_type == "player":
                # Check if velocity for y is near stationary
                if abs(obj.velocity[1]) <= FPS:
                    obj.velocity[1] = -JUMP_SPEED


def init():

    # List of objects
    obj_list = [
        fire(
            center=(10, 10),
            width=10, height=20, color=(255, 0, 0),
            obj_type="player",
            mass=1, acceleration=(0, GRAVITY)),
        block(
            center=(50, 70),
            width=100, height=6, color=(0, 200, 200),
            obj_type="background",
            mass=-1, acceleration=(0, 0))
    ]

    return obj_list, time.time()


def endscene(canvas, duration=2):

    # The transformation matrix
    cy = 0.5 * (canvas.shape[0] - 1)
    cx = 0.5 * (canvas.shape[1] - 1)
    T_inv = np.asarray([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]
    ])
    angle = 2 * np.pi / (duration * FPS)
    Rot = np.asarray([
        [np.cos(angle), np.sin(angle), 0.0],
        [-np.sin(angle), np.cos(angle), 0.0],
        [0, 0, 1]
    ])
    s = (0.1)**(1.0 / (duration * FPS))
    Scale = np.array([
        [s, 0, 0],
        [0, s, 0.],
        [0, 0, 1],
    ])
    T = np.asarray([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])
    delta_M = np.matmul(Rot, T_inv)
    delta_M = np.matmul(Scale, delta_M)
    delta_M = np.matmul(T, delta_M)

    # Disply cutscene
    M = np.eye(3)
    for i in range(int(duration * FPS)):

        new_canvas = canvas.copy()
        M = np.matmul(delta_M, M)
        new_canvas = cv2.warpAffine(
            new_canvas, M[:2], canvas.shape[:2], borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow(
            "canvas",
            cv2.resize(new_canvas, (500, 500), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(int(round(1000 / FPS)))


def gameplay(obj_list, canvas, init_time):

    new_obj_list = []

    # For all objects if health is zero, remove it from object list
    is_player_alive = False
    for obj in obj_list:
        if obj.health > 0:
            # For player
            if obj.obj_type == "player":
                is_player_alive = True
                new_obj_list += [obj]
            # For obstacle
            elif obj.obj_type == "obstacle":
                # Add only if object is in screen
                if obj.get_points()[:, 0].min() > 0:
                    new_obj_list += [obj]
            # For background
            else:
                new_obj_list += [obj]

    if not is_player_alive:
        # If player does not exist in the obj list, do cutscene and initialize
        # object list
        endscene(canvas)
        new_obj_list, init_time = init()

    else:
        # Otherwise, randomly add a red block
        prob_add = 1 / FPS
        if prob_add > np.random.rand():
            bottom = 66
            height = np.random.rand() * 30 + 10
            cy = (bottom - height + bottom) * 0.5
            new_obj_list += [
                block(
                    center=(100, cy),
                    width=5, height=height, color=(0, 0, 255),
                    obj_type="obstacle",
                    velocity=(-100, 0),
                    mass=-1, acceleration=(0, 0))
            ]

        pass

    return new_obj_list, init_time


def main():

    obj_list, init_time = init()

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

        # We now display the survival time
        cur_time = time.time() - init_time
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            canvas, str(int(round(cur_time))), (5, 10), font, 0.3, (0, 255, 0))

        # We'll compute gameplay logic here, just to simplify some
        # implementations, such as drawing :-) This is not perfect but should
        # not make much difference.
        obj_list, init_time = gameplay(obj_list, canvas, init_time)

        # Final display
        cv2.imshow(
            "canvas", cv2.resize(canvas, (500, 500), interpolation=cv2.INTER_NEAREST))

        # Set mouse callback for this window
        param = {"obj_list": obj_list}
        cv2.setMouseCallback("canvas", mouse_callback, param)

        # Deal with keyboard input
        if 113 == key:
            print("done")
            break
        elif 114 == key:
            obj_list, init_time = init()
            print("reset")

    pass


if __name__ == "__main__":
    main()
    exit(0)


#
# lecture10-collsion-detection-and-handling-part1.py ends here
