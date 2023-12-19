import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 

def cup_count(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    im_thresh = cv2.inRange(img_gray,180,255) 
    kernel = np.ones((1,15), np.uint8)
    im_erode =cv2.erode(im_thresh,kernel,iterations=1)
    sum_peaks = np.sum(im_erode,axis=1)
    # print(sum_peaks.shape)
    peaks,_ = sp.signal.find_peaks(sum_peaks, height=10000, distance=10)
    cup_count = len(peaks) - 1
    return cup_count 
def text_write(img,count):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,50)
    fontScale              = 3
    fontColor              = (255,0,0)
    thickness              = 3
    lineType               = 2
    cv2.putText(img,"{}".format(count),
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def main():
    cap = cv2.VideoCapture('demo2.mp4')
    if not cap.isOpened:
        print("Camera is not available. Exiting....")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("The frame is not reading from the camera.Exiting...")
            break
        img = frame[150:450,230:405,:]
        count = cup_count(img)
        print(count)
        frame = text_write(frame,count)
        cv2.imshow('frame',frame)
        if cv2.waitKey(5) == ord("q"):
            break 
    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
    