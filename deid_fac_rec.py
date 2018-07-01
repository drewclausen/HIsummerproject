from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

from StringIO import StringIO
import pandas as pd
import numpy as np
import pytesseract
import glob

import cv2
import future_builtins

import subprocess
import os
import sys
import re


import face_recognition


DIR = '/home/pikachu/PycharmProjects/deidentification/Scraped_Images/Faces/train/'
batch = 32
images= []
fnames = glob.glob(DIR + '*.jpg')


for fname in fnames:
    print fname
    fname_pre = fname.replace('train', 'train_pre')
    fname_deid = fname.replace('train', 'train_deid').replace('.','_deid.')

    #im = Image.open(fname)
    #image = face_recognition.load_image_file(fname)

    image = cv2.imread(fname)
    face_locations = face_recognition.face_locations(image)
    for face in face_locations:
        print face
        (top, right, bottom, left) = face

        face_image = image[top:bottom, left:right]

        # Blur the face image
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)

        # Put the blurred face region back into the frame image
        image[top:bottom, left:right] = face_image

        #im.paste(ic, box)
    cv2.imwrite(fname_deid, image)

