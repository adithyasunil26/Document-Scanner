# python scan.py --image images/page.jpg

import numpy as np
import argparse
import cv2
import imutils
from skimage.filters import threshold_local
from transform import four_point_transform,order_points
from PIL import Image

def show(image):
    cv2.imshow('', imutils.resize(image, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showcomp(a,b):
    cv2.imshow("Original", imutils.resize(a, height = 650))
    cv2.imshow("Scanned", imutils.resize(b, height = 650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save(args,warped):
    filename="scanned"+args
    cv2.imwrite(filename,warped)
    image1 = Image.open(r'{}'.format(filename))
    im1 = image1.convert('RGB')
    im1.save(r'scanned{}.pdf'.format(filename))

def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True,
        help = "Path to the image to be scanned")
    args = vars(ap.parse_args())
    return args

def loadimg(args):
    image = cv2.imread(args)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    return image,orig,ratio

def detectedge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    return gray,edged

def findcontour(image,edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    return image,screenCnt

def warping(orig,screenCnt):
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    return warped


args=argparser()["image"]
image,orig,ratio=loadimg(args)
gray,edged=detectedge(image)

show(image)
show(edged)

image,screenCnt=findcontour(image,edged)

show(image)

warped=warping(orig,screenCnt)

showcomp(orig,warped)

save(args,warped)