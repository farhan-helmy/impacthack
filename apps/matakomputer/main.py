import cv2 as cv
import sys

img = cv.imread(cv.samples.findFile("gambartest.jpeg"))
if img is None:
    sys.exit("Could not read the image.")
cv.imshow("Display window", img)
k = cv.waitKey(0)
if k == ord("s"):
    cv.imwrite("gambartest.jpeg", img)
