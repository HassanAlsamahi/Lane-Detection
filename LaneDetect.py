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

def SlidingWindows(binary_warped,nwindows = 9,widt = 100 ,minpix = 50):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((img,img,img))*255

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #Height of each window
    windows_height = np.int(binary_warped[0]//nwindows)
    #Identify x,y for the lanes line
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    #create empty lists to recieve the right and left lane pixels indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        window_y_low = binary_warped.shape[0] - (window+1)*windows_height
        window_y_high = binary_warped.shape[0] - window*window_height

        window_xleft_low = leftx_current - margin
        window_xleft_high = leftx_current + margin
        window_xright_low = rightx_current - margin
        window_xright_high = rightx_current + margin

        #Draw the windows of the left lane
        cv2.rectangle(out_img,(window_xleft_low,window_y_low),(window_xleft_high,window_y_high),(0,255,0),2)
        #Draw the windows of the right lane
        cv2.rectangle(out_img,(window_xright_low,window_y_low),(window_xright_high,window_y_high),(0,0,255),2)

        #Identify the nonzero values of x and y in each left window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high)).nonzero()[0]
        #Identify the nonzero values of x and y in each right window
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= window_xright_low) & (nonzerox < window_xright_high)).nonzero()[0]

        #Append this in left and right lanes
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        #EXtract left and right line pixels positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx,lefty,rightx,righty,out_img

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
