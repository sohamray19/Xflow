
# coding: utf-8

# In[15]:

import keras
import string
import pickle
import nltk
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
nltk.download('stopwords')
from flask import Flask,jsonify,request
from flask_restful import Resource, Api
from flask_restful import reqparse, abort, Api, Resource
def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = nltk.stem.SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text


# In[16]:

model_glove = keras.models.load_model('model.h5')
pickle_path = 'vec.pkl'
tok_pickle = open(pickle_path, 'rb')
tokenizer = pickle.load(tok_pickle)
app = Flask(__name__)
api = Api(app)

class Server(Resource):
    def post(self):
        evalSentence=request.get_json()
        sequences = tokenizer.texts_to_sequences([evalSentence['data']])
        data = pad_sequences(sequences, maxlen=50)
        pred = model_glove.predict(data)
        if pred[0][0]>0.5:
            buf = ("Positive Sentiment.Confidence Level:%d" % ((pred[0][0]-0.5)*200))
        else:
            buf = ("Negative Sentiment.Confidence Level:%d" % ((0.5-pred[0][0])*200))
        return jsonify({'Prediction': buf})


# In[17]:


api.add_resource(Server, '/')

if __name__ == '__main__':
    app.run(port=5003,debug=False)

