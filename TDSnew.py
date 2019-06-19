import numpy as np
import math
import scipy
from scipy import misc
from scipy import ndimage
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

def sobel(array):
	#print array.shape
        Gdash=array
        #print Gdash.shape
        shape=array.shape
        gx=np.array([[-2,-1,0,1,2] for i in range(0,5)])
        gy=np.array([np.repeat(i,5) for i in range (-2,3)])
        gf=np.array([[np.exp(-(gx[i,j]*gx[i,j]+gy[i,j]*gy[i,j])/3) for j in range(0,5)] for i in range(0,5)])
        smooth=np.array([([np.sum(gf*array[i:i+5,j:j+5])/25 for j in range(0,shape[1]-4)]) for i in range(0,shape[0]-4)])
        for i in range(2,shape[0]-2):
            for j in range(2,shape[1]-2):
                 Gdash[i,j]=smooth[i-2,j-2]
        G=Gdash
        #print G.shape
        sobely=np.array([[3,0,-3],[10,0,-10],[3,0,-3]])
        sobelx=np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
        Gx=np.array([([np.sum(sobelx*array[i:i+3,j:j+3])/9 for j in range(0,shape[1]-2)]) for i in range(0,shape[0]-2)])
        Gy=np.array([([np.sum(sobely*array[i:i+3,j:j+3])/9 for j in range(0,shape[1]-2)]) for i in range(0,shape[0]-2)])                
        Gx=[[Gx[i,j]*Gx[i,j] for j in range(0,shape[1]-2)] for i in range(0,shape[0]-2)]
        Gy=[[Gy[i,j]*Gy[i,j] for j in range(0,shape[1]-2)] for i in range(0,shape[0]-2)]
        tempG=np.sqrt(np.add(Gx,Gy))
        for i in range(1,shape[0]-1):
            for j in range(1,shape[1]-1):
                 G[i,j]=tempG[i-1,j-1]
        G.astype(int)      
        return G


def Area(img):
	naffect=0
	affect=0
	for rownum in range(len(img)):
		for colnum in range(len(img[rownum])):
			if img[rownum,colnum] == 0 :
				naffect=naffect+1
			else:
				affect=affect+1
	a=naffect-affect
	A=affect
	return a,A	

					
if __name__== "__main__":


    #   THRESHOLDING

    img=cv2.imread('test.jpg')
    imgORIG=img
    #img = misc.imread('test2.jpg')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image=img
    cv2.imshow("Input image",image)

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    cv2.imshow("Greyscale image",grey)

    ret, thresh = cv2.threshold(grey, 55, 255,  cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imshow("Thresholded/Binary image",thresh)

    grayTemp=thresh
    grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
    grayTemp = cv2.morphologyEx(grayTemp, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    erimg=grayTemp
    
    cv2.imshow("opened image",erimg)
    cv2.imwrite("threshold.jpg",thresh)
    cv2.imwrite("erimg.jpg",erimg)



    #Masking

    ret, mask = cv2.threshold(erimg, 10, 255, cv2.THRESH_BINARY)
    new_img = cv2.bitwise_and(image,image,mask=mask)
    
    cv2.imshow("FINAL image",new_img)
    cv2.imwrite("result.jpg",new_img)




    #ABCD

    img = misc.imread('result.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    imgColour = imgORIG

    erimg = misc.imread('erimg.jpg')

    cv2.imshow("Processed image",img)
    cv2.imshow("Closed image",erimg)

    #ASYMMETRY
    r=cv2.selectROI(imgORIG)
    imCrop = erimg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    a,A=Area(imCrop)
    a=abs(a)
    A=abs(A)
    print('a',a)
    print('A',A)
    Asymm=(a/A)
    
    if Asymm<0.25 :
        Asymmetry=2
    elif Asymm<0.60:
        Asymmetry=1
    else:
        Asymmetry=0
    print('Asymmetry')
    print(Asymmetry)

    img=cv2.resize(img,(135,135))
    erimg=cv2.resize(erimg,(135,135))

    #BORDER
    gigi=sobel(erimg)
    cv2.imshow("Gigi",gigi)

    #perimeter
    pcount=0
    for row in range(len(gigi)):
        for col in range(len(gigi[row])):
            if gigi[row,col] == 1:
                pcount=pcount+1
    P=pcount
    print(P)
    Border=(P/8)%8    
    #Border=((P*P)/(4*3.14*A))/10

    print('Border')
    print(Border)


    #DIAMETER
    n=(4*A)/P
    Diameter=(math.sqrt(n))/10
    Diameter=Diameter%5
    if Diameter==0:
        Diameter=5
    print('Diameter')
    print(Diameter)


    #COLOR
    colour=0
    # define the list of boundaries
    boundaries = [
    ([0, 0, 102], [51, 51, 255]),       #red
    ([63, 133, 205], [153, 204, 255]),  #light brown
    ([0, 25, 51], [44, 148, 220]),      #dark brown
    ([219, 219, 219], [255, 255, 255]), #white
    ([0, 0, 0], [44, 44, 44]),          #black
    ([101, 86, 45], [155, 133, 128])    #blue-grey
    ]
    # loop over the boundaries
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        mask = cv2.inRange(imgColour, lower, upper)
        #output = cv2.bitwise_and(image, image, mask = mask)
        count_white = np.sum(mask == 255)
        #print('Number of white pixels:', n_white_pix)
        if(count_white>100):
            colour=colour+1
        # show the images
        #cv2.imshow("mask",mask);
        #cv2.imshow("images", np.hstack([image, output]))
        #cv2.waitKey(0)

    print('Colour')
    print(colour)

    
    #TOTAL DERMOSCOPY RULE
    TDS = (Asymmetry*1.3) + (Border*0.1) + (colour*0.5) + (Diameter*0.5)
    print('TDS')
    print(TDS)
    if TDS<4.75:
        print('BENIGN MELACOCYTIC LESION')
    if TDS>=4.8 and TDS<=5.45:
        print('SUSPICIOUS LESION')
    if TDS>5.45:
        print('CANCEROUS MOLE')


    cv2.waitKey(0);

