import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ALLOWED_EXTENSIONS = set(["jpg","png","jpeg"])

def allowed_file(file):
    file = file.split('.')
    if file[1] in ALLOWED_EXTENSIONS:
        return True
    else: 
        return False

