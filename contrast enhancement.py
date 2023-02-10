import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
s = str(sys.argv[1])

img = cv2.imread(s,0)
s1='enhanced-'+s

#functions for luminescence i.e gamma correction
def increase_contrast(img,gamma,key):   
    if (key==2):
        img_enhanced=np.zeros((img.shape),dtype='uint8')
        for i in range (0,img.shape[0]):
            for j in range (0,img.shape[1]):
                img_enhanced[i][j]=apply_gamma(img[i,j],gamma)
                
    if (key==3):
        img_enhanced=np.zeros((img.shape),dtype='uint8')
        for i in range (0,img.shape[0]):
            for j in range (0,img.shape[1]):
                for k in range (0,img.shape[2]):
                    img_enhanced[i][j][k]=apply_gamma(img[i,j,k],gamma)
                
    return img_enhanced     

def apply_gamma(x,gamma):
    x1=float(x)
    inv=1.0/gamma
    x2=int(((x1/255.0)**(inv))*255.0)
    return x2

#functions for contrast enhancement using histogram
def get_histogram(img, bins):
    histogram = np.zeros(bins)
    for pixel in img:
        histogram[pixel] += 1
    return histogram

def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)

img = np.asarray(img)
# flattening the image
flat = img.flatten()
hist = get_histogram(flat, 256)
cums = cumsum(hist)
Nj = (cums - cums.min()) * 255
N = cums.max() - cums.min()
cums = Nj / N
cums = cums.astype('uint8')
img_new = cums[flat]

# put array back into original shape since we flattened it
img_new = np.reshape(img_new, img.shape)
img_output=increase_contrast(img_new,1.15,len(img_new.shape))
plt.imsave("./enhanced-"+s[2:],img_output,cmap='gray')
