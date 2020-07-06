#import pytesseract
from PIL import Image
import pytesseract
import argparse
from datetime import datetime

def OCR(lang='eng'):
    im = Image.open('images\ocr_1.jpg')
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    now = datetime.now()
    print(now)
    text = pytesseract.image_to_string(im, lang='kor')
    now = datetime.now()
    print('OCR')
    print(now)
    print(text)


OCR()
