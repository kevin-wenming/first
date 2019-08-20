#coding=utf8
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import sys
import cv2

faceClassifier = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)
fc = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    sys.stdout.write('\b'*30)
    image = frame.array
    tar = cv2.resize(image,(180,135),interpolation=cv2.INTER_CUBIC)
    cvtImage = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.1,minNeighbors=5)
    rawCapture.truncate(0)
    if (foundFaces!=()):
        print "a face at:%s" % foundFaces
        cv2.imwrite("testbak.jpg", image);
    sys.stdout.write('fc:%s' % fc)
    sys.stdout.flush()
    fc+=1
