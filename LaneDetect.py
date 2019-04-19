import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob



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

    src3 = np.float32([[130,342],[530,342],[347,228],[297,228]])
    dst3 = np.float32([[130,img.shape[0]],[530,img.shape[0]],[530,0],[130,0]])

    src4 = np.float32([[264,249],[384,249],[347,228],[297,228]])
    dst4 = np.float32([[264,img.shape[0]],[384,img.shape[0]],[384,0],[264,0]])

    Mtx = cv2.getPerspectiveTransform(src3,dst3)
    warped = cv2.warpPerspective(img,Mtx,img_size,cv2.INTER_LINEAR)

    return warped

def SlidingWindows(binary_warped,nwindows = 9,width = 100 ,minpix = 50):
    bottom_half = binary_warped[binary_warped.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255

    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

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

    return leftx,lefty,rightx,righty,out_img


def PolyFit(binary_warped):
    img = binary_warped.shape
    leftx,lefty,rightx,righty,out_img = SlidingWindows(binary_warped)
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

    return out_img

def search_around_poly(binary_warped,left_fit,right_fit,width = 100):
    # HYPERPARAMETER
    # Choose the width of the width around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    width = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- width of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - width)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + width)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - width)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + width)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = PolyFit(binary_warped)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+width,
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-width, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+width,
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result


video = cv2.VideoCapture("Vehicle Detection Raw Video.mp4")

#mtx,dist,img = CameraCal()


while video.isOpened():
    ret,frame = video.read()
    #undistort = cv2.undistort(frame,mtx,dist,None,mtx)


    Mask,s = threshold(frame,(100,255))
    transformed = PerspectiveTransform(Mask)
    output = PolyFit(transformed)




    cv2.imshow("Warped",transformed)
    cv2.imshow("Frame",frame)
    cv2.imshow("Mask",Mask)
    cv2.imshow("Result",output)

    #plt.imshow(output)
    #plt.show()


    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

video.release()
