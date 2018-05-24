from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
import pandas as pd
import pytesseract
import glob

elemsPerBox = 6

fnames = glob.glob('*.jpg')
for fname in fnames:
    print fname
    im = Image.open(fname)
    boxes = StringIO(pytesseract.image_to_data(im,config='--psm 12 --oem 3'))
    boxFrame = pd.read_table(boxes)
    for i in boxFrame.loc[(boxFrame.width > 0) & (boxFrame.height>0) & (boxFrame.conf>0)].itertuples():
        box = [i[7],i[8],i[7]+i[9],i[8]+i[10]]
        ic = im.crop(box)
        ic = ic.filter(ImageFilter.GaussianBlur(radius=200))
        im.paste(ic, box)
    im.save(fname.replace('.','_deid.'))
    im.close()
        