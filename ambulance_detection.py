from ultralytics import YOLO
import cv2 as cv

model = YOLO("best.pt")  # Load YOLOv8 model


result = model("input_images/ambulance.jpg", show = True)  # Inference on a single image

cv.waitKey(0)