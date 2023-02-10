import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
s = str(sys.argv[1])

img=cv2.imread(s,-1)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def gkernel(l, sig):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)
def apply_gkernel(img,l,sig):
    g=gkernel(l,sig)
    img1=cv2.filter2D(img,-1,sig)
    return img1

rgb_planes = cv2.split(img)


result_norm_planes = []
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = apply_gkernel(plane,25,10)
    bg_img = cv2.medianBlur(dilated_img, 25)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    result_norm_planes.append(norm_img)


result_norm = cv2.merge(result_norm_planes)
plt.imsave('cleaned-gutter.jpg',result_norm)