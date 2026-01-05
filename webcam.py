from ultralytics import YOLO
import cv2 as cv

model =YOLO('yolo-weights/yolo11n.pt')

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

while True:
    success, img = cap.read()
    results = model(img, show=True)
    cv.waitKey(1)