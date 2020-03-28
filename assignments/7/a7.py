import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import order_filter

TODO = None


def hough_transform(edges, num_r_bins, num_theta_bins):
    """Hough Transform.

    This function should return three things. The accumulator, the r values for
    each bin of the accumulator, the theta values for bins of the accumulator.

    """

    # Find all edge coordinates
    #
    # Behold the numpy magic!!! This function returns indices of where the
    # element is non-zero
    y, x = np.where(edges)
    x = x.reshape(-1, 1) - (edges.shape[1] - 1) / 2
    y = y.reshape(-1, 1) - (edges.shape[0] - 1) / 2

    # --------------------------------------------------------------------------
    # TODO: 3 marks: Create the accumulator. If you do this with python loops
    # you'll get three marks. If you want an extra mark, you can implement this 
    # using vectorized numpy functions for much better performance. Also have a 
    # look at `convert_to_r_theta` if you are unsure what you should return.

    acc = TODO
    rs = TODO
    thetas = TODO

    # --------------------------------------------------------------------------


    return acc, rs.flatten(), thetas.flatten()


def non_max_sup(acc):
    """Perform non-maximum suppression with a 3-by-3 neighborhood
    
    """

    # --------------------------------------------------------------------------
    # TODO: 3 marks. As before, if you use numpy functions instead of loops you 
    # will get a bonus mark. Hint: use the order_filter that I already imported. 
    # A local maximum is strictly larger than all of its neighborhood.

    nms_map = TODO

    # --------------------------------------------------------------------------

    return nms_map
    

def convert_to_r_theta(nms_map, rs, thetas):
    """ Gives the r, thetas in the nms map """

    r, theta = np.where(nms_map)

    return rs[r], thetas[theta]

def draw_line(img, r, theta):
    """ Draws a line using th r, theta representation """

    img_copy = img.copy()

    # r = x cos t + y sin t
    # y = - cost / sint * x + r / sint
    # x = - sint / cost * y + r / cost

    for _r, _theta in zip(r, theta):

        # _r = 7
        # print(_r, _theta * 180.0 / np.pi)
        # if _r > 8:
        #     continue

        cos = np.cos(_theta)
        sin = np.sin(_theta)

        if cos > sin:
            # this is when slope is large, we need to draw line based on y
            # extremes
            y_min = -(img.shape[0] - 1) / 2
            y_max = +(img.shape[0] - 1) / 2 
            x_min = _r / cos - sin / cos * y_min
            x_max = _r / cos - sin / cos * y_max
        else:
            # This is the opposite case
            x_min = -(img.shape[1] - 1) / 2 
            x_max = +(img.shape[1] - 1) / 2
            y_min = _r / sin - cos / sin * x_min
            y_max = _r / sin - cos / sin * x_max
        x_min += (img.shape[1] - 1) / 2 
        x_max += (img.shape[1] - 1) / 2 
        y_min += (img.shape[0] - 1) / 2 
        y_max += (img.shape[0] - 1) / 2 
        cv2.line(
            img_copy,
            (int(round(x_min)), int(round(y_min))),
            (int(round(x_max)), int(round(y_max))),
            (0,0,255), 1, cv2.LINE_AA)
            

    return img_copy


def main():

    # Read color image
    img = cv2.imread("input.jpg", 1)

    # We'll resize to a manageable size
    ds_rate = 8
    img = cv2.resize(img, (img.shape[1] // ds_rate, img.shape[0] // ds_rate))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find edges in images
    edges = cv2.Canny(img_gray, 200, 400)
    cv2.imwrite("edges.png", edges)

    # Get accumulator
    acc, rs, thetas = hough_transform(edges, 200, 200)

    # Save using matplotlib to look pretty
    plt.imsave("accumulator.png", acc)

    # Perform NMS map
    nms_map = non_max_sup(acc)
    plt.imsave("non_maximum_points.png", nms_map)

    # Select the top K points in the map to draw
    K = 10
    th = np.sort((nms_map * acc).flatten())[::-1][K]
    nms_map = nms_map * (acc > th)
    plt.imsave("top_k_points.png", nms_map)

    # Convert NMS map to r, theta
    r, theta = convert_to_r_theta(nms_map, rs, thetas)
    
    # Draw result
    res_img = draw_line(img, r, theta)
    cv2.imwrite("result.png", res_img)


if __name__ == "__main__":
    main()
    exit(0)