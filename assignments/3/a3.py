import time

import cv2
import numpy as np

FPS = np.array(60, dtype=np.float)
GRAVITY = 5431
CR = 0.5
DRAW_SIZE = 501
CANVAS_SIZE = 501
BALL_SIZE = 33
MAX_V = 2000

INIT_X = 50
INIT_Y = 50
CUR_X = 50
CUR_Y = 50
INIT_VX = 0
INIT_VY = 0
INIT_DO = True
MOUSE_DOWN = False

TODO = None

class block(object):
    """floor class"""

    def __init__(self,
                 center,
                 width,
                 height,
                 color,
                 mass,
                 acceleration=(0, 0),
                 velocity=(0, 0),
                 obj_type="block"):

        self.center = np.array(center, dtype=np.float)
        self.prev_center = np.array(center, dtype=np.float)
        self.updated = True
        self.color = color
        self.obj_type = obj_type

        # The *state* that we will change
        self.acceleration = np.array(acceleration, dtype=np.float)
        self.velocity = np.array(velocity, dtype=np.float)
        self.prev_velocity = np.array(velocity, dtype=np.float)
        self.mass = mass

        self.width = width
        self.height = height
        # the points determining this block, with the center at origin
        self.pts = np.array([
            [-(self.width - 1.0) * 0.5, -(self.height - 1.0) * 0.5],
            [+(self.width - 1.0) * 0.5, -(self.height - 1.0) * 0.5],
            [+(self.width - 1.0) * 0.5, +(self.height - 1.0) * 0.5],
            [-(self.width - 1.0) * 0.5, +(self.height - 1.0) * 0.5],
        ])

    def modify_velocity(self, dv):
        # ----------------------------------------------------------------------
        # TODO: 1 Mark: Function that modifies velocity safely.
        # The function should modify velocity according to dv, and check if it 
        # is within the allowed velocity limit, and clip to maximum velocity if
        # necessary. 

        TODO

        # ----------------------------------------------------------------------

    def update(self):
        # Store the state before updating
        self.prev_center = self.center.copy()
        self.prev_velocity = self.velocity.copy()
        # Update position
        self.center += (self.velocity / FPS)
        # Update velocity
        self.modify_velocity(self.acceleration / FPS)
        # Record update so that we don't recalculate
        self.updated = True

    def undo_update(self):
        # Undo velocity
        self.velocity = self.prev_velocity
        # Undo position
        self.center = self.prev_center
        # Record update so that we don't recalculate
        self.updated = True

    def redo_update(self, update_mask):
        # Redo position for the non-masked direction
        self.center += (1.0 - update_mask) * (self.velocity / FPS)
        # Redo velocity (do not mask as acc is always applied at each time
        # instance)
        self.modify_velocity(self.acceleration / FPS)
        # Record update so that we don't recalculate
        self.updated = True

    def get_points(self):

        if self.updated:
            self.abs_pts = np.round(self.center[None] - self.pts).astype(
                np.int)
            self.updated = False

        return self.abs_pts

    def draw(self, canvas):

        if self.obj_type == "block":
            cv2.fillConvexPoly(
                canvas,
                self.get_points()[None],
                self.color,
                lineType=cv2.LINE_AA)

        elif self.obj_type == "ball":
            # ------------------------------------------------------------------
            # TODO: 1 Mark: Draw the antialiased ball. Draw the ball on canvas,
            # where the ball should have AABB equal to the "block" case.

            TODO
            
            # ------------------------------------------------------------------

        else:
            raise ValueError("Wrong object type {}".format(self.obj_type))


def solve_direction(pair, obj_list):
    """
    Parameters
    ----------
    pair: integer tuple
        Index of colliding pair of objects in the `obj_list`.
    obj_list: list of objects
        List of objects in the scene.
    Returns
    -------
    mask: numpy array of size 2
        Return 1 on the dimension that should be updated with collision
        handling. For example, (1, 0) would mean that the collision is on
        horizontal direction only, and the vertical velocity and update should
        remain intact. If (0, 1) it should mean that only vertical direction
        should be updated. This was briefly explained in a Lecture.
    """
    # --------------------------------------------------------------------------
    # TODO: 3 Marks: Write a function that resolves the direction of collision.

    mask = np.array([0.0, 0.0])
    idx0, idx1 = pair

    # --------------------------------------------------------------------------

    return mask


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
        if cur_end >= next_start - 1:
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
        if cur_end >= next_start - 1:
            min_idx = min(y_sorted[i], y_sorted[i + 1])
            max_idx = max(y_sorted[i], y_sorted[i + 1])
            y_overlap.append((min_idx, max_idx))

    # For each overlap, check if the other direction also overlaps. It could
    # happen that they are not in both lists, due to the order the collision
    # detection is carried out.
    collision = []
    for overlap in x_overlap:
        # Only resolve collision for the ball
        if not (obj_list[overlap[0]].obj_type == "ball"
                or obj_list[overlap[1]].obj_type == "ball"):
            continue
        pts0 = obj_list[overlap[0]].get_points()[:, 1]
        pts1 = obj_list[overlap[1]].get_points()[:, 1]
        max_min = max(pts0.min(), pts1.min())
        min_max = min(pts0.max(), pts1.max())
        if min_max >= max_min:
            collision += [overlap]
    for overlap in y_overlap:
        # Only resolve collision for the ball
        if not (obj_list[overlap[0]].obj_type == "ball"
                or obj_list[overlap[1]].obj_type == "ball"):
            continue
        # Skip all that we already added to collision list
        if overlap not in collision:
            pts0 = obj_list[overlap[0]].get_points()[:, 0]
            pts1 = obj_list[overlap[1]].get_points()[:, 0]
            max_min = max(pts0.min(), pts1.min())
            min_max = min(pts0.max(), pts1.max())
            if min_max >= max_min:
                collision += [overlap]

    # Deal with collision
    for pair in collision[::-1]:
        # Check collision direction. We wil first have to check which direction
        # the collision happened to later on handle physics. You can check this
        # by doing some tests
        update_mask = solve_direction(pair, obj_list)

        # We will first undo our updates
        for idx in pair:
            obj = obj_list[idx]
            obj.undo_update()

        # Equation from https://en.wikipedia.org/wiki/Coefficient_of_restitution
        va_i = obj_list[pair[0]].velocity
        ma = obj_list[pair[0]].mass
        vb_i = obj_list[pair[1]].velocity
        mb = obj_list[pair[1]].mass

        # We'll use negative number for mass to denote infinite :-) Note that
        # this processing here will cause weird things to happen!
        if ma > 0 and mb > 0:
            va_f = (ma * va_i + mb * vb_i + mb * CR *
                    (vb_i - va_i)) / (ma + mb)
            vb_f = (ma * va_i + mb * vb_i + ma * CR *
                    (va_i - vb_i)) / (ma + mb)
        elif ma > 0:
            va_f = -CR * va_i
            vb_f = vb_i
        elif mb > 0:
            va_f = va_i
            vb_f = -CR * vb_i
        else:
            va_f = va_i
            vb_f = vb_i

        # Update only the correct direction. The other direction remains the
        # same.
        obj_list[pair[0]].modify_velocity(update_mask * (va_f - va_i))
        obj_list[pair[1]].modify_velocity(update_mask * (vb_f - vb_i))

        # Don't update position, simply apply acceleration
        for idx in pair:
            obj_list[idx].redo_update(update_mask)


# mouse callback function
def mouse_callback(event, x, y, flags, param):
    global INIT_X, INIT_Y, INIT_DO, INIT_VX, INIT_VY, CUR_X, CUR_Y, MOUSE_DOWN

    # Rescale x, y to canvas size
    x = x * CANVAS_SIZE / DRAW_SIZE
    y = y * CANVAS_SIZE / DRAW_SIZE

    if event == cv2.EVENT_LBUTTONDOWN:
        INIT_X = x
        INIT_Y = y
        INIT_VX = x
        INIT_VY = y
        MOUSE_DOWN = True
    if event == cv2.EVENT_LBUTTONUP:
        INIT_VX = (x - INIT_VX) * 3
        INIT_VY = (y - INIT_VY) * 3
        INIT_DO = True
        MOUSE_DOWN = False
    if event == cv2.EVENT_MOUSEMOVE:
        CUR_X = x
        CUR_Y = y


def init():

    # List of objects
    obj_list = []

    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Add the four Walls just outside of the canvas so that they
    # won't be drawn, but should make the ball bounce off the edges of the
    # screen. Note that collision is only detected when the ball is one of the
    # colliding objects. Thus, you don't have to worry about collision issues
    # between walls.



    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Shoot the ball according to the direction given by the
    # user. Modify the velocity and the center below.

    velocity = TODO
    center = TODO
    color = (255, 0, 0)
    obj_list += [
        block(
            center=center,
            width=BALL_SIZE,
            height=BALL_SIZE,
            color=color,
            mass=1,
            acceleration=(0, GRAVITY),
            velocity=velocity,
            obj_type="ball"),
    ]

    # --------------------------------------------------------------------------

    global INIT_DO
    INIT_DO = False

    return obj_list


def main():

    global INIT_DO

    while True:

        if INIT_DO:
            obj_list = init()

        # For the FPS counter
        time_start = time.time()

        canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
        # Draw things
        for obj in obj_list:
            obj.update()
        process_collision(obj_list)
        for obj in obj_list:
            obj.draw(canvas)

        # ----------------------------------------------------------------------
        # TODO: 1 Mark: Draw the arrow that points from the point where the
        # left mouse button was first clicked, to the current mouse cursor
        # position. Use thickness 5, and with anti-aliasing.

        if MOUSE_DOWN:
            TODO

        # ----------------------------------------------------------------------

        # Measure compute/rendering time
        time_comp = time.time() - time_start
        time_wait = 1000.0 / FPS - 1000.0 * time_comp
        key = cv2.waitKey(int(time_wait))

        # For the FPS counter
        cur_fps = 1.0 / (time.time() - time_start)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            canvas,
            str(int(round(cur_fps))), (5, 30),
            font,
            1, (0, 255, 0),
            lineType=cv2.LINE_AA)

        # Final display
        cv2.imshow(
            "canvas",
            cv2.resize(
                canvas, (DRAW_SIZE, DRAW_SIZE),
                interpolation=cv2.INTER_NEAREST))

        # Set mouse callback for this window
        cv2.setMouseCallback("canvas", mouse_callback)

        # Deal with keyboard input
        if 113 == key:
            print("done")
            break
        elif 114 == key:
            INIT_DO = True
            print("reset")


if __name__ == "__main__":
    main()
    exit(0)