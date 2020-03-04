import time

import cv2
import numpy as np
from numpy.fft import fftshift, fft2, ifft2

# Placeholder to stop auto-syntax checking from complaining
TODO = None

def main():

    # Read Image in grayscale and show
    img = cv2.imread("input.jpg", 0)
    cv2.imwrite("orig.png", img)

    # --------------------------------------------------------------------------
    # Create Filter
    # 
    # TODO: 3 Marks: Create sharpen filter from the lecture, but with a
    # Gaussian filter form the averaging instead of the mean filter. For the
    # Gaussian filter, use a kernel with size 31x31 with sigma 5. For the unit
    # impulse set the multiplier to be 2.

    # To get you started, here is a 1D Gaussian filter of size 31 and sigma=5
    filter1D = cv2.getGaussianKernel(31, 5)

    kernel = TODO

    # --------------------------------------------------------------------------

    # Filter with FFT
    # 
    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Pad filter with zeros to have the same size as the image,
    # but with the filter in the center. This creates a larger filter, that
    # effectively does the same thing as the original image.

    kernel_padded = TODO

    # --------------------------------------------------------------------------

    # Shift filter image to have origin on 0,0. This one is done for you. The
    # exact theory behind this was not explained in class so you may skip this
    # part.
    kernel_padded_shifted = fftshift(kernel_padded)

    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Move all signal to Fourier space (DFT).

    img_fft = TODO
    kernel_fft = TODO

    # --------------------------------------------------------------------------

    # Display signals in Fourier Space
    # I put some visualization here to help debugging :-)
    cv2.imwrite(
        "orig_fft.png",
        np.minimum(1e-5 * np.abs(fftshift(img_fft)), 1.0) * 255.)
    cv2.imwrite(
        "filt_fft.png",
        np.minimum(1e-1 * np.abs(fftshift(kernel_fft)), 1.0) * 255.)

    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Do filtering in Fourier space
    img_filtered_fft = TODO

    # --------------------------------------------------------------------------
    # TODO: 1 Mark: Bring back to Spatial domain (Inverse DFT)
    # TODO: 2 Marks: Throw away the imaginary part and clip between 0 and 255
    # to make it a real image.

    img_sharpened = TODO
    cv2.imwrite("res_fft.png", img_sharpened.astype(np.uint8))

    # --------------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # Filter with OpenCV
    # TODO: 1 Mark: Use padded filter and cyclic padding (wrap) to get exact results
    # TOOD: 1 Mark: Clip image for display

    img_sharpened = TODO

    # --------------------------------------------------------------------------

    cv2.imwrite("res_opencv.png", img_sharpened.astype(np.uint8))

    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
    exit(0)

#
# solution.py ends here
