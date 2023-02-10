



import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


#defining a simple canvas for all numbers where all possible circles are already present

def blank_canvas():
    img1=np.zeros((300,500))
    
    for i in range (0,3):
        for j in range (0,5):
            img1=cv2.circle(img1,(65+60*i,33+j*58),25,255,-1) 
 
    for i in range (0,3):
        for j in range (0,5):
            img1=cv2.circle(img1,(500-(65+60*i),33+j*58),25,255,-1)
            
    return img1

#function to modify the image according to need last depicts the position of number
def number(img,k,last):
    if (k==1):
        for i in range (0,3,2):
            for j in range (0,5):
                img=cv2.circle(img,((65+60*i)+last*250,33+j*58),25,0,-1)  
    if (k==0):
         for i in range (1,2):
            for j in range (1,4):
                img=cv2.circle(img,((65+60*i)+last*250,33+j*58),25,0,-1)
        
    if (k==3):
        img=cv2.circle(img,((65+60*0)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*0)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
    
    if (k==2):
        img=cv2.circle(img,((65+60*0)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*2)+last*250,33+3*58),25,0,-1)
        
    if (k==5):
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*2)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*0)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
        
    if (k==4):
        img=cv2.circle(img,((65+60*1)+last*250,33+0*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*0)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*0)+last*250,33+4*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+4*58),25,0,-1)
        
    if (k==6):
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*2)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
    
    if (k==7):
        for i in range(0,2):
            for j in range(1,5):
                img=cv2.circle(img,((65+60*i)+last*250,33+j*58),25,0,-1)
        img=cv2.circle(img,((65+60*0)+last*250,33+1*58),25,255,-1)      
    
    if (k==8):
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
    
    if (k==9):
        img=cv2.circle(img,((65+60*1)+last*250,33+1*58),25,0,-1)
        
        img=cv2.circle(img,((65+60*0)+last*250,33+3*58),25,0,-1)
        img=cv2.circle(img,((65+60*1)+last*250,33+3*58),25,0,-1)
        
        
    return img  

#takes the input and returns the output
n=int(sys.argv[1])
ans=blank_canvas()
for i in range (1,3):
    last1=i%2
    ans=number(ans,n%10,last1)
    n=int(n/10)
#plt.imshow(ans,cmap='gray')
plt.imsave('dotmatrix.jpg',ans,cmap='gray')







