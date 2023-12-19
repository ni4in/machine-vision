import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

def get_dims():
    ap = argparse.ArgumentParser("YOLOv8 Detection")
    ap.add_argument(
        "--webcam_resolution",
        nargs=2,
        type=int,
        help="Custom Resolution for Webcam"
        )
    args = ap.parse_args()
    return args 
    
def main():
    cap = cv2.VideoCapture('demo2.mp4')
    dims = get_dims()
    frame_height, frame_width = dims.webcam_resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    model = YOLO("yolov8l.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    if not cap.isOpened():
        print("cannot open the camera")
        exit()
    while True:
        ret,frame = cap.read()
        if not ret:
            print("Cant receive the frame from the camera. Exiting....")
            break 
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
        f"{model.model.names[class_id]}{confidence:0.2f}"
        for _,confidence,class_id,_ in detections
        ]
        frame = box_annotator.annotate(scene=frame.copy(), 
                            detections=detections,
                            labels=labels)

        cv2.imshow('webcam',frame)
        if cv2.waitKey(5) == ord("q"):
            break 
        
    cap.release()
    cv2.destroyAllWindows()



if __name__== "__main__":
    main()