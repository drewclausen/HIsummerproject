from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
import pandas as pd
import pytesseract
import glob
#afrom preprocess_image import preprocess
from PIL import ImageEnhance
import cv2
import future_builtins

import subprocess
import os
import sys
import re



elemsPerBox = 6


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


    ''' 
        python facial recognition tool
        check LFW dataset and current transfer learning nets
    '''


    #print COMMAND + image_file + " " + image_file_output
    os.system(COMMAND + image_file + " " + image_file_output)


def run_tesseract(im):
    return StringIO(pytesseract.image_to_data(im,config='--psm 11 --oem 3').encode("ascii","ignore"))

#fnames = glob.glob('/home/pikachu/PycharmProjects/deidentification/Scraped_Images/MSRA-TD500/train/*.JPG')
fnames = glob.glob('/home/pikachu/PycharmProjects/deidentification/Scraped_Images/KAIST/train/*.JPG')


for fname in fnames:
    fname_pre = fname.replace('train', 'train_pre')
    fname_deid = fname.replace('train', 'train_deid').replace('.','_deid.')

    im = Image.open(fname)

    # im = preprocess(im)
    # im.save(fname_pre)

    print fname, fname_pre

    script_preprocess(fname, fname_pre) #textprocessing script called


    im = Image.open(fname_pre)

    #print type(pytesseract.image_to_data(im,config='--psm 11 --oem 3')) #returns unicode or str (ascii) when include the .encode("ascii", "ignore")

    boxes = run_tesseract(im)
    im = Image.open(fname)

    try:
        boxFrame = pd.read_table(boxes, encoding = 'ascii')
        for i in boxFrame.loc[(boxFrame.width > 0) & (boxFrame.height>0) & (boxFrame.conf>0)].itertuples():
            box = [i[7],i[8],i[7]+i[9],i[8]+i[10]]
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

ARGUMENTS:

-r rotate ... ROTATE image either clockwise or counterclockwise by 90 degrees, if image aspect ratio does not match the layout mode. Choices are: cc (or clockwise), ccw (or counterclockwise) and n (or none). The default is no rotation.

-l layout ... LAYOUT for determining if rotation is to be applied. The choices are p (or portrait) or l (or landscape). The image will be rotated if rotate is specified and the aspect ratio of the image does not match the layout chosen. The default is portrait.

-c cropoffsets ... CROPOFFSETS are the image cropping offsets after potential rotate 90. Choices: one, two or four non-negative integer comma separated values. One value will crop all around. Two values will crop at left/right,top/bottom. Four values will crop left,top,right,bottom.

-g ... Convert the document to grayscale.

-e enhance ... ENHANCE brightness of image. The choices are: none, stretch, or normalize. The default=stretch.

-f filtersize ... FILTERSIZE is the size of the filter used to clean up the background. Values are integers>0. The filtersize needs to be larger than the thickness of the writing, but the smaller the better beyond this. Making it larger will increase the processing time and may lose text. The default is 15.

-o offset ... OFFSET is the offset threshold in percent used by the filter to eliminate noise. Values are integers>=0. Values too small will leave much noise and artifacts in the result. Values too large will remove too much text leaving gaps. The default is 5.

-t threshold ... THRESHOLD is the text smoothing threshold. Values are integers between 0 and 100. Smaller values smooth/thicken the text more. Larger values thin, but can result in gaps in the text. Nominal value is in the middle at about 50. The default is to disable smoothing.

-s sharpamt ... SHARPAMT is the amount of pixel sharpening to be applied to the resulting text. Values are floats>=0. If used, it should be small (suggested about 1). The default=0 (no sharpening).

-S saturation ... SATURATION is the desired color saturation of the text expressed as a percentage. Values are integers>=0. A value of 100 is no change. Larger values will make the text colors more saturated. The default=200 indicates double saturation. Not applicable when -g option specified.

-a adaptblur ... ADAPTBLUR applies an alternate text smoothing using and adaptive blur. The values are floats>=0. The default=0 indicates no blurring.

-u ... UNROTATE the image. This is limited to about 5 degrees or less.

-T ... TRIM the border around the image.

-p padamt ... PADAMT is the border pad amount in pixels. The default=0.

-b bgcolor ... BGCOLOR is the desired background color after it has been cleaned up. Any valid IM color may be use. If bgcolor=image, then the color will be computed from the top left corner pixel and a fuzzval. The final color will be computed subsequently as an average over the whole image. The default is white.

-F fuzzval ... FUZZVAL is the fuzz value for determining bgcolor when bgcolor=image. Values are integers>=0. The default=10.

-i invert ... INVERT colors for example to convert white text on black background to black text on white background. The choices are: 1 or 2 for one-way or two-ways (input or input and output). The default is no inversion.'''









