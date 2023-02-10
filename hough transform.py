import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
s = str(sys.argv[1])
shapes = cv2.imread(s)
#shapes=cv2.cvtColor(shapes, cv2.COLOR_BGR2RGB)

def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape 
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) 
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))
    H = np.zeros((len(rhos)+10, len(thetas)+10), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) 

    for i in range(len(x_idxs)): 
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas

def hough_peaks(H, num_peaks, threshold, nhood_size):
    indicies = []
    H1 = np.copy(H)
    for i in range(num_peaks):
        idx = np.argmax(H1) 
        H1_idx = np.unravel_index(idx, H1.shape) 
        indicies.append(H1_idx)

        idx_y, idx_x = H1_idx
        if (idx_x - (nhood_size/2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size/2)
        if ((idx_x + (nhood_size/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (nhood_size/2) + 1

      
        if (idx_y - (nhood_size/2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size/2)
        if ((idx_y + (nhood_size/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size/2) + 1
            
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                H1[y, x] = 0
                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255
    return indicies, H

def hough_lines_draw(img, indicies, rhos, thetas):
    img1=img.copy()
    c=0
    for i in range(len(indicies)):
        c=c+1
        # reverse engineer lines from rhos and thetas
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        # these are then scaled so that the lines go off the edges of the image
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img1,c

im1=cv2.cvtColor(shapes, cv2.COLOR_BGR2RGB)
shapes_grayscale = cv2.cvtColor(shapes, cv2.COLOR_RGB2GRAY)
shapes_blurred = cv2.GaussianBlur(shapes_grayscale, (5, 5), 1.5)
canny_edges = cv2.Canny(shapes_blurred, 50, 120)

H, rhos, thetas = hough_lines_acc(canny_edges)
indicies, H = hough_peaks(H,12, 1000,10) # 1000,80
v,c=hough_lines_draw(im1, indicies, rhos, thetas)
plt.imsave("./robolin-"+s[2:],v)