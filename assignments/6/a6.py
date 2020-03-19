import time

import cv2
import numpy as np
from numpy.fft import fftshift, fft2, ifft2

NUM_BIN = 32
NUM_COLOR = 16
TODO = None


def quantize_colors(img, colors):
    """Quantizes colors to the nearest one"""

    # Conver colors into numpy array to do magic
    colors = np.array(colors)

    # Reshape image
    img_f = np.reshape(img, (-1, 3))

    # Compute distances from each pixel to the color. Note that I use
    # broadcasting.
    dists = img_f.reshape((-1, 1, 3)) - colors.reshape(1, -1, 3)
    # square and add slong the last dimension (i.e. the rgb)
    dists = np.sum(dists**2, axis=-1)

    # Find the one with the smallest distance
    bests = np.argmin(dists, axis=1)

    # Now use these colors instead. Behold the numpy magic. We are now treating
    # the bests as an array holding indices, which what it is, and then using
    # that to recreate a matrix by indexing colors with this index
    # operation. Note that each indexing retrieves 3 elements as color is of
    # shape Nx3. Thus if bests was Mx1, we get Mx3 matrix at the end. This is
    # quite advanced so if you are interested you can come talk to me.
    img_new = colors[bests]

    # Reshape into the original images shape
    img_new = np.reshape(img_new, img.shape)

    return img_new


def populousity(img, num_colors=16):
    """Populousity algorithm"""

    # Vectorize image so that we can easily create histogram using
    # numpy.histogramdd
    img_f = np.reshape(img, (-1, 3))

    # Create an histogram to find the most prominant colors but with only
    # NUM_BINxNUM_BINxNUM_BIN bins for efficiency
    hist, bins = np.histogramdd(
        img_f, bins=[NUM_BIN, NUM_BIN, NUM_BIN],
        range=[[0, 255], [0, 255], [0, 255]])

    # Find the populous colors
    colors = []
    for _ in range(num_colors):
        # ----------------------------------------------------------------------
        # TODO: 1 Mark: Find the color that is most dominant by looking at the
        # histogram. Comment the random color line after you are done
        # implementing. The random line is just to give you an idea of how the
        # code runs.

        color_populous = TODO
        color_populous = (np.random.rand(3)*NUM_BIN).astype(np.int)

        # ----------------------------------------------------------------------

        # Store that (as original colors by multiplying)
        colors += [np.array(color_populous) * 256 / NUM_BIN]
        # We now set that value to zero so that we find the next populous
        hist[color_populous] = 0

    # Apply the quantization outcome
    return quantize_colors(img, colors)


class box:

    def __init__(self, level, hist, ranges):
        """Initializes the box"""

        # Remember the level of split
        self.level = level
        # Remember histogram
        self.hist = hist
        # Remember the ranges (b, g, r order)
        self.ranges = np.array(ranges)

        # Compute the color of this box form the histogram and the range
        # values. Note that for efficiency we use the histogram. We will also
        # set so that the indices start and end points can be easilly used for
        # indexing. e.g. box that is [[0, 32], [0, 32], [0. 16]] will include
        # colors that fall into the histogram bin of (0, 0, 0), (31, 20, 15),
        # but not (0, 10, 16). Similar to the Python convention of array
        # indexing and slicing.
        #
        # One more thing is that histogram bins correspond to different color
        # values. We can also opt to go back to the original color value when
        # we compute this representative color for the box, but here for
        # simplicities sake we will just do a weighted sum using the histogram
        # counts.
        #
        # This part seemed too hard and I did it for you :-)
        self.sub_hist = self.hist[
            self.ranges[0][0]:self.ranges[0][1],
            self.ranges[1][0]:self.ranges[1][1],
            self.ranges[2][0]:self.ranges[2][1],
        ]
        self.sub_hist_colors = np.array(np.meshgrid(
            np.arange(self.ranges[0][0], self.ranges[0][1]),
            np.arange(self.ranges[1][0], self.ranges[1][1]),
            np.arange(self.ranges[2][0], self.ranges[2][1]),
            indexing="ij",
        )).transpose(1, 2, 3, 0)
        # Weigh each color bin color with histogram count
        self.color = self.sub_hist[..., None] * self.sub_hist_colors
        # Do weighted average (make sure the denom is larger than 1)
        num_pixel = self.sub_hist.sum()
        self.color = self.color.sum(axis=(0, 1, 2)) / max(1, num_pixel)
        # Convert color into 256 scale
        self.color *= 256 / NUM_BIN

        # Also record the number of pixels in this box
        self.num_pixel = num_pixel

        # Also remember that if a box can be split (might have all elements in
        # one bin!)
        self.is_splitable = True

    def split(self):
        """Splits the boxes into two"""

        # ----------------------------------------------------------------------
        # TODO: 1 Mark: Sort the dimensions according to the length of the cube
        # in **decreasing** order. We will try in this order, since if a split
        # creates an empty box, it's useless and we'll try the next dimension.
        # Incase the direction is identical in all dimensions, simply choose in
        # the order of r, g, b. (2, 1, 0)

        sorted_idx = TODO

        # ----------------------------------------------------------------------

        for max_idx in sorted_idx:

            # ------------------------------------------------------------------
            # Split along that direction, by looking at the histogram and
            # choosing the bin that is at the median. N

            # TODO: 1 Mark: Marginalize the sub histogram, i.e. create a
            # temporary histogram that is the sum of all other dimensions
            # except for the one in interest

            temp_hist = TODO

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # TODO: 1 Mark: To find the median, do a cumulative sum and then
            # search for the bin that just passes 0.5. That's the bin holding
            # the median value.

            med_idx = TODO


            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Split the range so that one starts from med_idx

            # First create a copy. Note that we explicitly copy since python
            # does call by reference by default, and would not really create a
            # copy.
            ranges1 = self.ranges.copy()
            ranges2 = self.ranges.copy()

            # TODO: 1 Mark: Now fix the ranges so that the block covers the
            # split halves. Note that we used the sub histograms to get the
            # median index, thus we need to convert it to the range in the
            # original histogram. We need to add the start point of the
            # ranges. Also note that we want the first range to cover until the
            # median.
            
            TODO

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # TODO: 1 Mark: Create split boxes and check if both are valid
            # return the two boxes

            box1 = box(TODO, self.hist, ranges1)
            box2 = box(TODO, self.hist, ranges2)

            # ------------------------------------------------------------------

            # Assertion to make sure you are splitting correctly
            assert(box1.num_pixel + box2.num_pixel == temp_cum_hist[-1])

            if box1.num_pixel > 0 and box2.num_pixel > 0:
                return box1, box2

        # Return None indicating that this box cannot be split
        self.is_splitable = False
        return None, None


def find_box_to_split(boxes):
    """Find the box to split"""

    box_to_split = None

    # For easy debugging, this function at first returns that no box is
    # splitable. This is to help you run the code as is at the beginning. You
    # can comment the first return line to avoid that from happening and
    # implement your code.
    return None

    # The recommended order of implementation is to first implement the split
    # so that you understand the box class. Thus the below two lines are added
    # so that you first split the first item in the list, without looking into
    # others. Comment this line and implement below after the other parts are
    # working.
    box_in_question = boxes[0]
    return box_in_question

    # ----------------------------------------------------------------------
    # Find box at the lowest level, with the larges number of pixels, that can
    # be split

    # Find splitable boxes
    splitable_boxes = [_box for _box in boxes if _box.is_splitable]
    if len(splitable_boxes) == 0:
        return None
    
    # TODO: 1 Mark: Find boxes with lowest level
    
    low_level_boxes = TODO
    
    # ----------------------------------------------------------------------

    # Find the box with the larges number of pixels with lowest level
    num_pixels = [_box.num_pixel for _box in low_level_boxes]
    idx_box = np.argmax(num_pixels)
    box_in_question = low_level_boxes[idx_box]

    # Check if the box can still split and simply return None indicating we no
    # longer have boxes to split here. We can do so by checking the box range
    # for the box in question and looking at the maximum range. If the maximum
    # range is below 1 we cannot split.
    ranges = box_in_question.ranges
    max_range = np.max(ranges[:, 1] - ranges[:, 0])
    if max_range > 1:
        box_to_split = box_in_question

    return box_to_split


def median_cut(img, num_colors=16):

    # Initialization

    # Vectorize image so that we can easily create histogram using
    # numpy.histogramdd
    img_f = np.reshape(img, (-1, 3))

    # Create an histogram to find the most prominant colors but with only
    # NUM_BINxNUM_BINxNUM_BIN bins for efficiency
    hist, bins = np.histogramdd(
        img_f, bins=[NUM_BIN, NUM_BIN, NUM_BIN],
        range=[[0, 255], [0, 255], [0, 255]])

    # Create a color box with the entire set of colors. .
    boxes = [box(0, hist, [[0, NUM_BIN], [0, NUM_BIN], [0, NUM_BIN]])]

    # Now the quantization part

    # Until we have all the colors (max num_color x 10 tries)
    for i in range(num_colors * 10):
        # Find the box to split
        box_to_split = find_box_to_split(boxes)
        # If we have a box to split
        if box_to_split is not None:
            # Split the box into two
            box1, box2 = box_to_split.split()
            if box1 is not None and box2 is not None:
                # Remove the box we split from the list
                boxes.remove(box_to_split)
                # Add the two split boxes to our list of boxes
                boxes = [box1, box2] + boxes
            # If the number of colors reached break.
            if len(boxes) >= num_colors:
                break
        else:
            break

    # Return quantized image by first converting the boxes into list of
    # representative colors and the applying quantization
    colors = [box.color for box in boxes]
    return quantize_colors(img, colors)

def main():

    # Read color image
    img = cv2.imread("input.jpg", 1)

    # Populousity algorithm
    img_pop = populousity(img, NUM_COLOR)
    cv2.imwrite("populousity.png", img_pop)

    # Median Cut algorithm
    img_medcut = median_cut(img, NUM_COLOR)
    cv2.imwrite("median_cut.png", img_medcut)


if __name__ == "__main__":
    main()
    exit(0)
