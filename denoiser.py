import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

s = str(sys.argv[1])


def gaussian(val,sigma):
    calc=np.exp(-0.5 * val / sigma**2)
    return calc

def dist_gaussian(centre,image,sigma_r):
    val=(centre - image)**2
    calc=np.exp(-0.5 * val / sigma_r**2)
    return calc

def bilateral_filtering(image,filter_size, sigma_d, sigma_r):
    image = padding(image)
    
    #used to store result and image difference
    weight_sum = np.zeros(image.shape)
    res = np.zeros(image.shape)
    
    for i in range(-filter_size, filter_size+1):
        for j in range(-filter_size, filter_size+1):
            
            # weights calculated for distance
            spatial_wght = gaussian(i**2 + j**2, sigma_d)
            
            #use to calculate the difference od whole image matrix
            #uses the concept of sliding the matrix
            off = np.roll(image, [i, j], axis=[0, 1])
            
            bilateral_f = spatial_wght * dist_gaussian( off, image, sigma_r)
            res += off*bilateral_f
            weight_sum += bilateral_f
    result=res / weight_sum
    return result

def padding(image):
    shape = image.shape
    zeros_h = np.zeros(shape[1]).reshape(-1, shape[1])
    zeros_v = np.zeros(shape[0]+2).reshape(shape[0]+2, -1) 
    padded_img = np.vstack((zeros_h, image, zeros_h)) # add rows
    padded_img = np.hstack((zeros_v, padded_img, zeros_v)) # add cols
    image = padded_img
    shape = image.shape
    return image  

filter_size=15
sigma=25
sigma_r1=25

if s=='noisy2.jpg':
    filter_size=9
    sigma_r1=12
    sigma=5

print("This code takes about 90 seconds to run for larger image")
img=cv2.cvtColor(cv2.imread(s), cv2.COLOR_BGR2RGB)
rgb_planes = cv2.split(img)
result_norm_planes = []
for plane in rgb_planes:
    norm_img=bilateral_filtering(plane,filter_size,sigma,sigma_r1)
    result_norm_planes.append(norm_img)
result_norm = cv2.merge(result_norm_planes)
z=(np.floor(result_norm)).astype(int)
z1=np.array(z,dtype='uint8')
plt.imsave("denoised.jpg",z1)