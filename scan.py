# python scan.py --image images/page.jpg

import numpy as np
import argparse
import cv2
import imutils
from skimage.filters import threshold_local
from utils import show,showcomp,save,argparser,loadimg
from imageprocessing import detectedge,findcontour,warping

def main():
    args=argparser()["image"]
    image,orig,ratio=loadimg(args)
    gray,edged=detectedge(image)

    show(image)
    show(edged)

    image,screenCnt=findcontour(image,edged)

    show(image)

    warped=warping(orig,screenCnt,ratio)

    showcomp(orig,warped)

    save(args,warped)

if __name__=="__main__":
    main()