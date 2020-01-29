import cv2
import numpy as np


# Not nice, but we'll just use some global variables
canvas = None
mouseX = -1
mouseY = -1
click = False
dblclick = False

# This will be a list of all the polygons currently on the screen.
# Each polygon is a list of points [x, y]. (list of lists of lists)
poly_pt_list = []

# This function is called every time the mouse does something
def mouse_callback(event, x, y, flags, params):
    """Mouse callback function"""

    global canvas, mouse_pt_list, click, dblclick, mouseX, mouseY

    if event == cv2.EVENT_MOUSEMOVE:
        mouseX = x
        mouseY = y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        dblclick = True
    elif event == cv2.EVENT_LBUTTONUP:
        click = True


def init():
    global poly_pt_list
    poly_pt_list = [[[mouseX, mouseY]]]


# We use this guard to allow functions from this script to be imported in other
# scripts without this code running.
if __name__ == "__main__":

    # Create named window
    window_name = "Press (r) to reset, (q) to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Initialize mouse
    init()

    # --------------------------------------------------------------------------
    # TODO: Load the lines of `input.txt` into a list (1 mark)

    lines = []

    # --------------------------------------------------------------------------

    # This is our event loop. This will run many times per second, redrawing the
    # canvas each time, as well as responding to any user input.
    while True:
        # Deal with mouse input
        if dblclick:
            # ------------------------------------------------------------------
            # This block will execute when the user double-clicks
            # TODO: Respond to this event by starting a new polygon. (1 mark)



            # ------------------------------------------------------------------
            dblclick = False
            click = False
        elif click:
            # ------------------------------------------------------------------
            # This block will execute when the user single-clicks
            # TODO: Respond to this event by adding a new point to the polygon
            # at the current mouse location. (1 mark)



            # ------------------------------------------------------------------
            click = False
        else:
            pass
            # ------------------------------------------------------------------
            # This block will execute when the user did not click during a frame
            # TODO: Update the position of the last point in the polygon so that
            # it follows the mouse. (1 mark) 

            

            # ------------------------------------------------------------------
        
        # ----------------------------------------------------------------------
        # TODO: Set the canvas size based on `input.txt`. (1 mark)
        # Hint: use some of your assignment 1 code here.

        canvas = np.ones((200, 200), dtype=np.float32)

        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # TODO: Draw the shapes from the lines of `input.txt`. (4 marks)
        # Hint: use some of your assignment 1 code here.

        cv2.circle(canvas,
                   color=(0, 0, 0),
                   center=(10, 10),
                   radius=5,
                   thickness=2)

        cv2.line(canvas,
                 color=(0, 0, 0),
                 pt1=(40, 10),
                 pt2=(10, 40),
                 thickness=2)

        cv2.fillPoly(canvas,
                     np.array([
                         [[40, 40], [40, 60], [60, 60], [60, 40]],
                         [[70, 70], [70, 80], [80, 80], [80, 70]]
                     ]),
                     color=(0, 0, 0))

        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # TODO: Draw the polygon from `poly_pt_list` (1 mark)



        # ----------------------------------------------------------------------

        # Show canvas
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10)

        # Deal with keyboard input
        if ord("q") == key:
            break
        elif ord("r") == key:
            init()
