import numpy as np
import cv2
lower_blue = np.array([100,150,0])
upper_blue = np.array([140,255,255])
I=cv2.imread('white.jpg')
e=cv2.imread('E.jpg')
fe=cv2.flip(e,1)
g=cv2.imread('G.jpg')
fg=cv2.flip(g,1)
b=cv2.imread('B.jpg')
fb=cv2.flip(b,1)
a=255
b=0
c=255
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
   
    cv2.rectangle(frame,(0,0),(0+200,0+100),(255,255,255),5)    #defining colour boxes at the top of frame
    cv2.rectangle(frame,(220,100),(220+200,0+100),(0,255,0),5)
    cv2.rectangle(frame,(440,100),(440+200,0+100),(0,0,0),5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv,lower_blue ,upper_blue )
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    kernel = np.ones((5,5),np.uint8)                                                           #morphological operations
    erosion=cv2.erode(mask, kernel, iterations =1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]                   #find shapes
    cv2.imshow('o/p1',mask)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]                          
    for rect in rects:
        
        if rect[2]>40 and rect[3]>40:
           cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(255,0,0),2)           #draw rectangles on target object
           xc=int(rect[0]+rect[2]/2)
           yc=int(rect[1]+rect[2]/2)
           print(rect[0],rect[1],rect[2])
           cv2.putText(frame,'.',(xc,yc),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),10)
           if rect[2]<200:
               if(rect[1]>=0 and rect[1]<=100):
                   if(rect[0]>=0 and rect[0]<=200):
                       a=255
                       b=255
                       c=255
                   elif(rect[0]>=220 and rect[0]<=420):
                       a=0
                       b=255
                       c=0
                   elif(rect[0]>=440 and rect[0]<=640):
                       a=0
                       b=0
                       c=0
               if(rect[1]<400):
                   if(a==255 and b==255 and c==255):
                       cv2.putText(I,'.',(xc,yc),cv2.FONT_HERSHEY_SIMPLEX,0.5,(a,b,c),100)
                   else:
                    cv2.putText(I,'.',(xc,yc),cv2.FONT_HERSHEY_SIMPLEX,0.5,(a,b,c),20)
    #cv2.drawContours(frame,cnts,-1,(0,255,0),2)
    fframe=cv2.flip(frame,1)
    cv2.imshow('frame',fframe)
    fimg=cv2.flip(I,1)
    cv2.imshow('drawing',fimg)
    k = cv2.waitKey(10)
    if k==27:
          break
h=fframe.shape[0]
w=fframe.shape[1]
print(h,w)
cv2.imwrite("draw1.jpg",fimg)
cv2.destroyAllWindows()
cap.release()
