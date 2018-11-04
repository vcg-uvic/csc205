
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

#Helper functions

def gkern(k=5, sigma=1.):
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))

    return kernel / np.sum(kernel)

#Extract from the image a section the same size as the kernel
def neighbors(r,c,A,ksize=3): 
    return A[r:r+ksize,c:c+ksize] 

#Perform the convolution
def convolve(n,kernel):
    return (kernel*n).sum()
    
"""
img = input grayscale image
kernel = is the kernel we're applying to the image
padding = what kind of padding around the image do we want to use (see Lecture notes)
"""
def imfilter2d(img,kernel,padding=cv2.BORDER_REPLICATE):
    #make sure kernel is square and odd-sized
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1
    
    #For convolution, an easy way to do the calculation is to flip the kernel and perform
    #cross-correlation
    conv_kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel for convolution
    
    ksize = kernel.shape[0]
    pad = kernel.shape[0] // 2
    result = np.ndarray(img.shape,dtype=np.float32)
    
    #Add padding to the image (if we don't we get an image smaller than our input!)
    img_pad = cv2.copyMakeBorder(img, top=pad,bottom=pad,left=pad,right=pad, borderType=padding)       
    

    #(r,c) is the center pixel
    for (r,c),value in np.ndenumerate(img):
        #Our padded is naturally offset which makes the extraction easier
        n = neighbors(r,c,img_pad,ksize)
        #Perform the convolution and save the result in the center pixel :)
        result[r,c] = convolve(n,conv_kernel)

    #What to do if the result is outside of 0-255?
    #We will get a range outside of 0-256. So we normalize between 0-255
    #and then use equalizeHist
    #This doesn't appear to be the technique that the OpenCV filter2d is using
    #but it's close enough for a demo
    result = (result - result.min()) / (result.max() - result.min())*255
    return cv2.equalizeHist(np.uint8(result))

imlena = cv2.imread('lena.png', 0)
fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(imlena, cmap='gray')
ax[1].imshow(cv2.equalizeHist(imlena), cmap='gray')


# In[2]:


#Apply an averaging fitler
ksize=11
kernel = np.ones((ksize,ksize), dtype=np.float32) / (ksize*ksize)


avg1= imfilter2d(imlena,kernel)

#Alternatively, use cv2.blur() or cv2.medianBlur()
avg2 = cv2.filter2D(imlena, #input image
             -1, #image depth, -1 uses the input image depth
             kernel, #kernel for convolution
             borderType=cv2.BORDER_REPLICATE) #How to pad the outside of the image so original image size is kept

fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(avg1, cmap='gray')
ax[1].imshow(avg2, cmap='gray')


# In[5]:


#Apply a gaussian filter
ksize=11
kernel = gkern(ksize, 5)


blur1= imfilter2d(imlena,kernel)

#Alternatively, use cv2.GaussianBlur()
blur2 = cv2.filter2D(imlena, #input image
             -1, #image depth, -1 uses the input image depth
             kernel, #kernel for convolution
             borderType=cv2.BORDER_REPLICATE) #How to pad the outside of the image so original image size is kept

fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(blur1, cmap='gray')
ax[1].imshow(blur2, cmap='gray')


# In[6]:


#TODO: Add your code
# Activity 1: Sharpen the image by creating your own kernel and applying it

sharpen = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]], dtype=np.float32)

imsharp = cv2.filter2D(imlena, #input image
             -1, #image depth, -1 uses the input image depth
             sharpen, #kernel for convolution
             borderType=cv2.BORDER_REPLICATE) #How to pad the outside of the image so original image size is kept

fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(imlena, cmap='gray')
ax[1].imshow(imsharp, cmap='gray')

# Activity 2: Add two or subtract filters together and apply them to the image
ksize=11
kernel1 = gkern(ksize, 5)
kernel2 = gkern(ksize, 11)
kernel = kernel2 - kernel1
imedge = cv2.filter2D(imlena, #input image
             -1, #image depth, -1 uses the input image depth
             kernel, #kernel for convolution
             borderType=cv2.BORDER_REPLICATE) #How to pad the outside of the image so original image size is kept

fig, ax = plt.subplots(1, 2, figsize=(15,15))
ax[0].imshow(imlena, cmap='gray')
ax[1].imshow(imedge, cmap='gray')
plt.show()
