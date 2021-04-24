from attention import AttentionLayer


import numpy as np  
import pandas as pd 
import re           
from bs4 import BeautifulSoup 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords   
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

f = open('training_dataset/train.ast.src', 'r')
input_data = str(f.readline())
# print(input_data)
# tokenizer_inputs = Tokenizer(num_words=len(input_data), filters='')
# tokenizer_inputs.fit_on_texts(input_data)
f.close()
data = input_data

stop_words = set(stopwords.words('english'))
stop_words.remove('if')
stop_words.remove('for')
stop_words.remove('this')

