import numpy as np
import argparse
import cv2
import imutils
from PIL import Image

def loadimg(args):
    image = cv2.imread(args)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    return image,orig,ratio
    
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