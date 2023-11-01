import cv2
import keypoints
import cameracali

window_name = "main_window"

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)

two_d, three_d = keypoints.calculate(calibrationImg)

K = cameracali.calculate(two_d, three_d)

