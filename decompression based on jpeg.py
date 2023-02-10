import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
s = str(sys.argv[1])
s2 = str(sys.argv[2])
s3 = str(sys.argv[3])

img_Y=cv2.imread(s,0)
img_cr4=cv2.imread(s3,0)
img_cb4=cv2.imread(s2,0)

stretch_near_img_cr4 = cv2.resize(img_cr4, (img_Y.shape[1],img_Y.shape[0]),
               interpolation = cv2.INTER_AREA)
stretch_near_img_cb4 = cv2.resize(img_cb4, (img_Y.shape[1],img_Y.shape[0]),
               interpolation = cv2.INTER_AREA)

img=[]
img.append(img_Y)
img.append(stretch_near_img_cb4)
img.append(stretch_near_img_cr4)
x1=cv2.merge(img)

def ycbcr2rgb(im):
    arr = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(arr.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

img_before_GU=ycbcr2rgb(x1)
def bilateralfilter(image, texture, sigma_s, sigma_r):
    r = int(np.ceil(3 * sigma_s))
    # Image padding
   
    h, w, ch = image.shape
    I = np.pad(image, ((r, r), (r, r), (0, 0)), 'symmetric').astype(np.float32)
    
    ht, wt = texture.shape
    T = np.pad(texture, ((r, r), (r, r)), 'symmetric').astype(np.int32)
    
    # Pre-compute
    output = np.zeros_like(image)
    scaleFactor_s = 1 / (2 * sigma_s * sigma_s)
    scaleFactor_r = 1 / (2 * sigma_r * sigma_r)
    # A lookup table for range kernel
    LUT = np.exp(-np.arange(256) * np.arange(256) * scaleFactor_r)
    # Generate a spatial Gaussian function
    x, y = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
    kernel_s = np.exp(-(x * x + y * y) * scaleFactor_s)
    
    
    # Main body    
    for y in range(r, r + h):
        for x in range(r, r + w):
            wgt = LUT[abs(T[y - r:y + r + 1, x - r:x + r + 1] - T[y, x])] * kernel_s
            wacc = np.sum(wgt)
            output[y - r, x - r, 0] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 0]) / wacc
            output[y - r, x - r, 1] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 1]) / wacc
            output[y - r, x - r, 2] = np.sum(wgt * I[y - r:y + r + 1, x - r:x + r + 1, 2]) / wacc

    return output



sigma_s = 7
sigma_r = 0.1
img_bf = bilateralfilter(img_before_GU, img_Y, sigma_s, sigma_r)
plt.imsave('flyingelephant.jpg',img_bf)