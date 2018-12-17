
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2
import numpy as np
import pandas as pd
import pickle
from keras.callbacks import EarlyStopping
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from json import dumps
sess = tf.Session()
print(tf.__version__)


# In[2]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import nltk
nltk.download('stopwords')
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE


# In[3]:

# Path to dataset
df =pd.read_csv('/home/exacon02/Desktop/XSent/Sentiment Analysis Dataset.csv',nrows=10000, sep=',',usecols=['SentimentText', 'Sentiment'],error_bad_lines=False)




df= df.dropna()
df = df[df.Sentiment.apply(lambda x: x !="")]
df = df[df.SentimentText.apply(lambda x: x !="")]

labels=df['Sentiment']
pickle_path = 'vec.pkl'
tok_pickle = open(pickle_path, 'wb')
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['SentimentText'])
sequences = tokenizer.texts_to_sequences(df['SentimentText'])
# print(sequences)
data = pad_sequences(sequences, maxlen=50)
pickle.dump(tokenizer, tok_pickle)
model = Sequential()
model.add(Embedding(20000, 128, input_length=50))
model.add(LSTM(128,dropout=0.5,return_sequences=True))
model.add(LSTM(128,dropout=0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
model.fit(data, np.array(labels),
          batch_size=512,
          epochs=20,
          verbose=1,
          validation_split=0.2,
          callbacks = [EarlyStopping(monitor='val_loss', patience=1)],
          shuffle=True)


# In[32]:


model.save("model.h5")
#
# # In[6]:
#
#
# def clean_text(text):
#
#     ## Remove puncuation
#     text = text.translate(string.punctuation)
#
#     ## Convert words to lower case and split them
#     text = text.lower().split()
#
#     ## Remove stop words
#     stops = set(stopwords.words("english"))
#     text = [w for w in text if not w in stops and len(w) >= 3]
#
#     text = " ".join(text)
#     ## Clean the text
#     text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
#     text = re.sub(r"what's", "what is ", text)
#     text = re.sub(r"\'s", " ", text)
#     text = re.sub(r"\'ve", " have ", text)
#     text = re.sub(r"n't", " not ", text)
#     text = re.sub(r"i'm", "i am ", text)
#     text = re.sub(r"\'re", " are ", text)
#     text = re.sub(r"\'d", " would ", text)
#     text = re.sub(r"\'ll", " will ", text)
#     text = re.sub(r",", " ", text)
#     text = re.sub(r"\.", " ", text)
#     text = re.sub(r"!", " ! ", text)
#     text = re.sub(r"\/", " ", text)
#     text = re.sub(r"\^", " ^ ", text)
#     text = re.sub(r"\+", " + ", text)
#     text = re.sub(r"\-", " - ", text)
#     text = re.sub(r"\=", " = ", text)
#     text = re.sub(r"'", " ", text)
#     text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
#     text = re.sub(r":", " : ", text)
#     text = re.sub(r" e g ", " eg ", text)
#     text = re.sub(r" b g ", " bg ", text)
#     text = re.sub(r" u s ", " american ", text)
#     text = re.sub(r"\0s", "0", text)
#     text = re.sub(r" 9 11 ", "911", text)
#     text = re.sub(r"e - mail", "email", text)
#     text = re.sub(r"j k", "jk", text)
#     text = re.sub(r"\s{2,}", " ", text)
#     ## Stemming
#     text = text.split()
#     stemmer = nltk.stem.SnowballStemmer('english')
#     stemmed_words = [stemmer.stem(word) for word in text]
#     text = " ".join(stemmed_words)
#     return text
#
#
# # In[7]:
#
#
# df['SentimentText'] = df['SentimentText'].map(lambda x: clean_text(x))
#
#
# # In[8]:
#
#
# labels=df['Sentiment']
#
#
# # In[17]:
#
#
# pickle_path = '/home/exacon02/git/KerasPOC-SentimentAnalysis/vec.pkl'
# tok_pickle = open(pickle_path, 'wb')
#
#
# # In[9]:
#
#
# # sel_pickle_path = '/home/exacon03/Jupyter/sel_pickle.pkl'
# # sel_pickle = open(sel_pickle_path, 'wb')
# vocabulary_size = 20000
# tokenizer = Tokenizer(num_words= vocabulary_size)
# tokenizer.fit_on_texts(df['SentimentText'])
# sequences = tokenizer.texts_to_sequences(df['SentimentText'])
# # print(sequences)
# data = pad_sequences(sequences, maxlen=50)
#
#
# # In[18]:
#
#
# pickle.dump(tokenizer, tok_pickle)
#
#
# # In[10]:
#
#
# embeddings_index = dict()
# f = open('/home/exacon02/glove.6B.100d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
#
# # In[11]:
#
#
# embedding_matrix = np.zeros((vocabulary_size, 100))
# for word, index in tokenizer.word_index.items():
#     if index > vocabulary_size - 1:
#         break
#     else:
#         embedding_vector = embeddings_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[index] = embedding_vector
#
#
# # In[ ]:
#
#
# model_glove = Sequential()
# model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
# model_glove.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# model_glove.add(Dense(1, activation='sigmoid'))
# model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ## Fit train data
# model_glove.fit(data, np.array(labels),
#   batch_size=512,
#   epochs=20,
#   verbose=1,
#   validation_split=0.2,
#   callbacks = [EarlyStopping(monitor='val_loss', patience=1)],
#   shuffle=True)
#
#
# # In[16]:
#
#
# model_glove.save("/home/exacon02/git/KerasPOC-SentimentAnalysis/plug.h5")



