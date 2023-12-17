import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cannot open the camera")
    exit()

while True:
    ret,frame = cap.read()
    
    if not ret:
        print("Cant receive the frame from the camera. Exiting....")
        break 
    cv2.imshow('webcam',frame)
    if cv2.waitKey(0) == ord("q"):
        break 
    
cap.release()
cv2.destroyAllWindows()
