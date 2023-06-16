import numpy as np
import cv2
import cvlib as cv  
  
# Capturing video through webcam
webcam = cv2.VideoCapture(2)
  
# Start a while loop
while(1):
      
    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()
  
    imageFrame = cv2.resize(imageFrame, (1020, 600))
    bbox, label, conf = cv.detect_common_objects(imageFrame, enable_gpu=True)

    output_image = cv.object_detection.draw_bbox(imageFrame, bbox, label, conf)

    # Program Termination
    cv2.imshow("Object detection", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break