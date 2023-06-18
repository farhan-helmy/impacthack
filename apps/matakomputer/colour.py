import numpy as np
import cv2
import pika
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from ulid import ULID

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='127.0.0.1', heartbeat=30))
channel = connection.channel()

channel.queue_declare(queue='amazonggo')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
session = boto3.Session(profile_name='impacthack')
s3_client = session.client('s3')

# Capturing video through webcam
webcam = cv2.VideoCapture(2)
webcam2 = cv2.VideoCapture(4)
prev_red_num = 0
prev_face_count = 0
# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()

    # Read the frame
    _, imageFrame2 = webcam2.read()
    # Convert to grayscale
    gray = cv2.cvtColor(imageFrame2, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_count = 0

    for (x, y, w, h) in faces:
        if w > 100 and h > 100:
            face_count += 1
            cv2.rectangle(imageFrame2, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('FaceDetection', imageFrame2)

    if face_count != prev_face_count:  # Publish only if face_count changes
        current_time = datetime.now().strftime(
            '%Y-%m-%d%H:%M:%S')  # Get current local time
        ulid = ULID()
        img_name = f"amazonggo_{str(ulid)}.jpeg"
        
        cv2.imwrite(img_name, imageFrame2)
        print("{} written!".format(img_name))
        print(f'Face count: {face_count} at {current_time}')
        try:
            response = s3_client.upload_file(img_name, 'amazonggoimpacthack', img_name)
            print(str(ulid))
            # channel.basic_publish(exchange='', routing_key='amazonggo',
            #                   body=f'https:')
        except ClientError as e:
            print(e)
        prev_face_count = face_count  # Update the previous face_count value

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernel = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernel)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    items = 4
    red_num = 0
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            red_num += 1
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, f"Red Colour {red_num}", (x, y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        (0, 0, 255))

    if red_num != prev_red_num:  # Publish only if red_num changes
        current_time = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')  # Get current local time
        channel.basic_publish(exchange='', routing_key='amazonggo',
                              body=f'Nasi lomak stock: {red_num} at {current_time}')
        prev_red_num = red_num  # Update the previous red_num value

    # Program Termination
    cv2.imshow("Amazonggo", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        webcam2.release()
        cv2.destroyAllWindows()
        connection.close()
        break
