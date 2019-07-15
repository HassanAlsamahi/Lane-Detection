import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math

def CameraCal():
    Photos_for_calibration_path = ""
    images = glob.glob(Photos_for_calibration_path)

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

    return final,s_channel,ScaledSobel

def PerspectiveTransform(img):
    img_size = (img.shape[1],img.shape[0])
    src = np.float32([[113,535],[824,535],[566,363],[394,363]])
    dst = np.float32([[113,535],[824,535],[824,0],[113,0]])

    src2 = np.float32([[160,355],[426,355],[350,288],[275,288]])
    dst2 = np.float32([[160,355],[426,355],[426,0],[160,0]])

    src3 = np.float32([[168,336],[555,333],[347,228],[297,228]])
    dst3 = np.float32([[168,img.shape[0]],[555,img.shape[0]],[555,0],[168,0]])      #Source and destination points of video 3
    #plt.plot(168,336, marker = 'x', color = 'red')
    #plt.plot(555,333 , marker = 'x', color = 'red')
    #plt.plot(347,228, marker = 'x', color = 'red')
    #plt.plot(285,231, marker = 'x', color = 'red')
    src4 = np.float32([[264,249],[384,249],[347,228],[297,228]])
    dst4 = np.float32([[264,img.shape[0]],[384,img.shape[0]],[384,0],[264,0]])

    Mtx = cv2.getPerspectiveTransform(src3,dst3)
    warped = cv2.warpPerspective(img,Mtx,img_size,cv2.INTER_LINEAR)

    return warped

def SlidingWindows(binary_warped,nwindows = 9,width = 100 ,minpix = 50, safe_pixels = 20):
    direction = ''
    bottom_half = binary_warped[binary_warped.shape[0]//2:,:]
    top_half = binary_warped[:binary_warped.shape[0]//2,:]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #Identify the direction of the lanes
    left_Start = leftx_base
    right_start = rightx_base
    image_histogram = np.sum(top_half,axis = 0)
    max_hist = np.argmax(image_histogram[:midpoint])
    right_max_hist = np.argmax(image_histogram[midpoint:])
    if max_hist > left_Start + safe_pixels:
        direction = 'right'
    elif max_hist < left_Start - safe_pixels:
        direction = 'Left'
    else:
        direction = "Straight"

    #Height of each window
    windows_height = np.int(binary_warped.shape[0]//nwindows)

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
        window_y_high = binary_warped.shape[0] - window*windows_height

        window_xleft_low = leftx_current - width
        window_xleft_high = leftx_current + width
        window_xright_low = rightx_current - width
        window_xright_high = rightx_current + width

        #Draw the windows of the left lane
        cv2.rectangle(out_img,(window_xleft_low,window_y_low),(window_xleft_high,window_y_high),(0,255,0),2)
        #Draw the windows of the right lane
        cv2.rectangle(out_img,(window_xright_low,window_y_low),(window_xright_high,window_y_high),(0,0,255),2)

        #Identify the nonzero values of x and y in each left window
        good_left_inds = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_xleft_low) & (nonzerox < window_xleft_high)).nonzero()[0]
        #Identify the nonzero values of x and y in each right window
        good_right_inds = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_xright_low) & (nonzerox < window_xright_high)).nonzero()[0]


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

    return leftx,lefty,rightx,righty,out_img,direction


def PolyFit(binary_warped):
    img = binary_warped.shape
    leftx,lefty,rightx,righty,out_img,direction = SlidingWindows(binary_warped)

    #Find our Lane Lines pixels
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, img[0]-1, img[0] )

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img,left_fit,right_fit,ploty,direction

def Measure_curvature(left,right,ploty,wheel_base=2.7,steering_ratio=16):
    #direction = ''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    angle = 0
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!


    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curve_rad = ((1 + (2*left[0]*y_eval*ym_per_pix + left[1])**2)**1.5) / np.absolute(2*left[0])
    right_curve_rad = ((1 + (2*right[0]*y_eval*ym_per_pix + right[1])**2)**1.5) / np.absolute(2*right[0])
    mean_curved = (left_curve_rad + right_curve_rad)/2
    angle = 8 * (180 / (math.pi)) * math.acos((2-(wheel_base/mean_curved)**2) / 2 )
    #theta = (steering_ratio/2) * math.acos(1- (wheel_base**2/2*mean_curved**2))



    '''
    #Direction of rotating
    if left_Slope > right_Slope:
        direction = 'Right'
    else:
        direction = 'Left'
    '''

    return left_curve_rad, right_curve_rad,angle

Vid1 = 'testvid.mp4'
Vid2 = 'test2.mp4'
Vid3 = 'Vehicle Detection Raw Video.mp4'

video = cv2.VideoCapture(Vid3)

#mtx,dist,img = CameraCal()


while video.isOpened():
    ret,frame = video.read()
    #undistort = cv2.undistort(frame,mtx,dist,None,mtx)


    Mask,s,sobelx = threshold(frame,(100,255))
    transformed = PerspectiveTransform(Mask)
    output,left_curve,right_curve, ploty, direction = PolyFit(transformed)
    left_lane_radius, right_lane_radius, angle = Measure_curvature(left_curve,right_curve,ploty)



    cv2.imshow("Warped",transformed)
    cv2.imshow("edges",sobelx)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",Mask)
    cv2.imshow("Result",output)
    cv2.imwrite("Edges.png",sobelx)

    print("Left Lane Curve: {:.4f} meters".format(left_lane_radius))
    print("Right Lane Curve: {:.4f} meters".format(right_lane_radius))
    print("Angle To rotate: {:.4f} degrees".format(angle))
    #print("theta: {:.4f}".format(theta))
    print("Direction: ",direction)
    print("----------------------------------------------------")


    '''
    plt.imshow(output)
    plt.show()

    state = input()
    if state == "x":
        break
    '''

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
