from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance

from StringIO import StringIO
import pandas as pd
import pytesseract
import glob

#from preprocess_image import preprocess
import cv2
import future_builtins

import subprocess
import os
import sys
import re


elemsPerBox = 6
DIR = '/home/pikachu/PycharmProjects/deidentification/Scraped_Images/KAIST/train/'

def preprocess(im, grayscale=False, edge_enhance=False):
    sharpen = ImageEnhance.Contrast(im)
    im = sharpen.enhance(factor=1.5)

    sharpen = ImageEnhance.Sharpness(im)
    im = sharpen.enhance(factor=1.5)

    if grayscale:
        im = im.convert('L')  # grayscale

    if edge_enhance:
        im = im.filter(ImageFilter.EDGE_ENHANCE) #seemed to make less accurate almost always
        im = im.filter(ImageFilter.EDGE_ENHANCE_MORE)

    im = im.filter(ImageFilter.MedianFilter())  # a little blur
    im = im.point(lambda x: 0 if x < 140 else 255)  # threshold (binarize)

    return im;



def script_preprocess(image_file, image_file_output):
    #1. COMMAND = "./textcleaner -e stretch -f 25 -o 20 -t 30 -s 1 "
    #2. COMMAND = "./textcleaner -e stretch -f 20 -o 10 -t 30 -s 1 "
    COMMAND = "./textcleaner -e stretch -f 20 -o 10 -t 30 -s 1 "

    #print COMMAND + image_file + " " + image_file_output
    os.system(COMMAND + image_file + " " + image_file_output)

    '''

    works pretty well 
        failures when:
        --- light text on dark background,
        --- pattern on background
        --- complex and colorful background
        --- fairly blurry or distorted text 

    Mostly underestimates but also may overestimate when strange background
    Also not meant for commercial purposes.
    '''




def run_tesseract(im):
    return StringIO(pytesseract.image_to_data(im,config='--psm 11 --oem 3').encode("ascii","ignore"))

#fnames = glob.glob('/home/pikachu/PycharmProjects/deidentification/Scraped_Images/MSRA-TD500/train/*.JPG')

fnames = glob.glob(DIR + '*.JPG')


for fname in fnames:
    fname_pre = fname.replace('train', 'train_pre')
    fname_deid = fname.replace('train', 'train_deid').replace('.','_deid.')

    #im = Image.open(fname)

    # im = preprocess(im)
    # im.save(fname_pre)

    print fname, fname_pre

    script_preprocess(fname, fname_pre) #textprocessing script called


    im = Image.open(fname_pre)

    #print type(pytesseract.image_to_data(im,config='--psm 11 --oem 3')) #returns unicode, or str (ascii) when append .encode("ascii", "ignore")

    boxes = run_tesseract(im)
    im = Image.open(fname)

    try:
        boxFrame = pd.read_table(boxes, encoding = 'ascii')
        for i in boxFrame.loc[(boxFrame.width > 0) & (boxFrame.height>0) & (boxFrame.conf>0)].itertuples():
            box = [i[7],i[8],i[7]+i[9],i[8]+i[10]]
            print box
            ic = im.crop(box)
            ic = ic.filter(ImageFilter.GaussianBlur(radius=200))
            im.paste(ic, box)
        im.save(fname.replace('train', 'train_deid').replace('.','_deid.'))
    except:
        print 'error'
        im.close()



#tenatively 10:30 - 3:00 Saturday the 16th




'''    
5/29/18
manually review images through the pipeline
    grouping with shadow or aligned different ways or such
    raw, proprocessing, see what works
    certain preprocessing formula
    enough categories then try classification
    go through some of the images and find what preprocessing works
    few weeks
'''

''' 
6/3/18
Tried multiple different encodings and decodings from ascii and unicode
On forums there is a solution for the python3 version, but no working solution for python2 (python3 solution not applicable for some reason)

Error: pandas.errors.ParserError: Error tokenizing data. C error: EOF inside string starting at line 40 
Not exactly sure why reaching end of the file. Happens when call read_table. 
Happens after most preprocessing steps even after saving and calling preprocessed image. 

Any help or advice would be greatly appreciated. 

Thank you so much,
Suraj
'''



'''
6/5/18
Tried using the imagemagik textcleaner linux script with very useful results in some situations. Worked better than 
most other previously tried methods




USAGE: textcleaner [-r rotate] [-l layout] [-c cropoff] [-g] [-e enhance ] [-f filtersize] [-o offset] [-u] [-t threshold] [-s sharpamt] [-s saturation] [-a adaptblur] [-T] [-p padamt] [-b bgcolor] [-F fuzzval] [-i invert] infile outfile
USAGE: textcleaner [-help]

-r .... rotate .......... rotate image 90 degrees in direction specified if 
......................... aspect ratio does not match layout; options are cw 
......................... (or clockwise), ccw (or counterclockwise) and n 
......................... (or none); default=none or no rotation
-l .... layout .......... desired layout; options are p (or portrait) or 
......................... l (or landscape); default=portrait
-c .... cropoff ......... image cropping offsets after potential rotate 90; 
......................... choices: one, two or four non-negative integer comma 
......................... separated values; one value will crop all around; 
......................... two values will crop at left/right,top/bottom; 
......................... four values will crop left,top,right,bottom
-g ...................... convert document to grayscale before enhancing
-e .... enhance ......... enhance image brightness before cleaning;
......................... choices are: none, stretch or normalize; 
......................... default=none
-f .... filtersize ...... size of filter used to clean background;
......................... integer>0; default=15
-o .... offset .......... offset of filter in percent used to reduce noise;
......................... integer>=0; default=5
-u ...................... unrotate image; cannot unrotate more than 
......................... about 5 degrees
-t .... threshold ....... text smoothing threshold; 0<=threshold<=100; 
......................... nominal value is about 50; default is no smoothing
-s .... sharpamt ........ sharpening amount in pixels; float>=0; 
......................... nominal about 1; default=0
-S .... saturation ...... color saturation expressed as percent; integer>=0; 
......................... only applicable if -g not set; a value of 100 is 
......................... no change; default=200 (double saturation)
-a .... adaptblur ....... alternate text smoothing using adaptive blur; 
......................... floats>=0; default=0 (no smoothing)
-T ...................... trim background around outer part of image 
-p .... padamt .......... border pad amount around outer part of image;
......................... integer>=0; default=0
-b .... bgcolor ......... desired color for background or "image"; default=white
-F .... fuzzval ......... fuzz value for determining bgcolor when bgcolor=image; 
......................... integer>=0; default=10
-i .... invert .......... invert colors; choices are: 1 or 2 for one-way or two-ways
......................... (input or input and output); default is no inversion

PURPOSE: To process a scanned document of text to clean the text background.

DESCRIPTION: TEXTCLEANER processses a scanned document of text to clean the text background and enhance the text. The order of processing is: 
1) optional 90 degree rotate if aspect does not match layout
2) optional crop, 
3) optional convert to grayscale, 
4) optional enhance, 
5) filter to clean background and optionally smooth/antialias, 
6) optional unrotate (limited to about 5 degrees or less), 
7) optional text smoothing,
8) optional sharpening, 
9) optional saturation change (if -g is not specified), 
10) optional alternate text smoothing via adaptive blur
11) optional auto trim of border (effective only if background well-cleaned),
12) optional pad of border
'''


