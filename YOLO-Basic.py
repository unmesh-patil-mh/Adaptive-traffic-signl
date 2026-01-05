from ultralytics import YOLO
import cv2 as cv

model = YOLO('best.pt')
# results = model('input_images/lane1.jpg', show=True)
# results2 = model('input_images/lane2.jpg', show=True)
# results3 = model('input_images/lane3.jpg', show=True)
results4 = model('input_images/ambulance.jpg', show=True)
cv.waitKey(0)