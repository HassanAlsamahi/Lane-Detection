import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def CameraCal():
    images = glob.glob("D:\hassan work\Programming Projects\Python projects\OpenCV projects\Intro to CV udacity course\Camera Calibration\GOPR00*.jpg")

    objPoints = [] #3D points in world
    imgPoints = [] #2D points in image plane

    objP = np.zeros((6*8,3),np.float32)
    objP[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

    for fname in images:

        img = cv2.imread(fname)

        #convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        #Find the chess boards corners
        ret,corners = cv2.findChessboardCorners(gray,(8,6),None)

        if ret is True:
            imgPoints.append(corners)
            objPoints.append(objP)
            cv2.drawChessboardCorners(img,(8,6),corners,ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints,imgPoints,gray.shape[::-1],None,None)

    return mtx,dist,img

def threshold(img,ColorThreshold = (120,250),GradientThreshold = (80,255),KernelSize=15):
    #convert the colors to hls
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    #convert the colors to gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Color Threshold mask
    binaryColor = np.zeros_like(s_channel)
    binaryColor[(s_channel > ColorThreshold[0]) & (s_channel <= ColorThreshold[1])] = 1

    #Sobel operator
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,KernelSize)
    abs_sobel = np.absolute(sobelx)
    ScaledSobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    #Gradient Threshold mask
    binarySobel = np.zeros_like(gray)
    binarySobel[(ScaledSobel > GradientThreshold[0]) & (ScaledSobel <= GradientThreshold[1])] = 1

    #Combined Threshold mask
    binary = np.zeros_like(gray)
    binary[(binaryColor == 1) | (binarySobel == 1)] = 1
    final = binary * 255

    return final,s_channel

def PerspectiveTransform(img):
    img_size = (img.shape[1],img.shape[0])
    src = np.float32([[113,535],[824,535],[566,363],[394,363]])
    dst = np.float32([[113,535],[824,535],[824,0],[113,0]])

    src2 = np.float32([[160,355],[426,355],[350,288],[275,288]])
    dst2 = np.float32([[160,355],[426,355],[426,0],[160,0]])


    Mtx = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,Mtx,img_size,cv2.INTER_LINEAR)

    return warped



video = cv2.VideoCapture("test2.mp4")

#mtx,dist,img = CameraCal()

while video.isOpened():
    ret,frame = video.read()
    #undistort = cv2.undistort(frame,mtx,dist,None,mtx)

    Mask,s = threshold(frame,(100,255))
    transformed = PerspectiveTransform(Mask)


    cv2.imshow("Warped",transformed)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",Mask)

    """
    plt.plot(113,535,"*")
    plt.plot(824,535,"*")
    plt.plot(394,363,"*")
    plt.plot(565,363,"*")
    plt.imshow(frame)
    #plt.plot(out_img)
    plt.show()
    """

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
"""
print(mtx)
print(dist)
cv2.imshow("image",img)
cv2.waitKey(0)
"""
video.release()
