import cv2

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")

window_name = "Calibration Image"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Displaying the image
cv2.imshow(window_name, calibrationImg)

# Waits for keypress to prevent instant closing
cv2.waitKey(0)

# Close all open windows
cv2.destroyAllWindows()
