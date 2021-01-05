from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from scipy.sparse import hstack
import nltk

app = Flask(__name__)


analyser=open('SentimentAnalyser.pkl','rb')
sid=pickle.load(analyser)
analyser.close()


def remove_htmltags(s):
    res = re.sub('<.*?>','',s)
    return res

def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase

def preprocess(text):
    # convert all the text into lower letters
    # use this function to remove the contractions: https://gist.github.com/anandborad/d410a49a493b56dace4f814ab5325bbd
    # remove all the spacial characters: except space ' '
    text = text.lower()
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text

def get_sentiment(row, **kwargs):
    sentiment_score = sid.polarity_scores(row)
    if kwargs['k'] == 'positive':
        return sentiment_score['pos']
    if kwargs['k'] == 'negative':
        return sentiment_score['neg']
    if kwargs['k'] == 'neu':
        return sentiment_score['neu']
    if kwargs['k'] == 'compound':
        return sentiment_score['compound']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        reviewtext = request.form['reviewtext']
        reviews=reviewtext
        reviewtext=  remove_htmltags(reviewtext)
        reviewtext = preprocess(reviewtext)
        
        reviewtext_pos = get_sentiment(reviewtext, k='positive')
        Norm_pos=open('Norm_pos.pkl','rb')
        normalizer=pickle.load(Norm_pos)
        Norm_pos.close()
        
        X_positive_norm = normalizer.transform(np.array(reviewtext_pos).reshape(1,-1))
        
        reviewtext_neg = get_sentiment(reviewtext, k='negative')
        
        Norm_neg=open('Norm_neg.pkl','rb')
        normalizer=pickle.load(Norm_neg)
        Norm_neg.close()
        
        X_negative_norm = normalizer.transform(np.array(reviewtext_neg).reshape(1,-1))
        
        reviewtext_neu = get_sentiment(reviewtext, k='neutral')
        

        reviewtext_neu=0
        
        
        Norm_neu=open('Norm_neu.pkl','rb')
        normalizer=pickle.load(Norm_neu)
        Norm_neu.close()
        
        X_neutral_norm = normalizer.transform(np.array(reviewtext_neu).reshape(1,-1))
        
        reviewtext_com = get_sentiment(reviewtext, k='compound')
        
        Norm_com=open('Norm_com.pkl','rb')
        normalizer=pickle.load(Norm_com)
        Norm_com.close()
        
        X_compund_norm = normalizer.transform(np.array(reviewtext_com).reshape(1,-1))
        
        vectorizer=open('Vectorizer.pkl','rb')
        vectorizer_tf=pickle.load(vectorizer)
        vectorizer.close()
        X_tfidf = vectorizer_tf.transform([reviewtext])
        #data = np.array([[reviewtext]])
        
        X_tf=hstack((X_tfidf,X_positive_norm,X_negative_norm,X_neutral_norm,X_compund_norm)).tocsr()
        
        log=open('classifier.pkl','rb')
        classifier=pickle.load(log)
        log.close()
        
  
        
        my_prediction = classifier.predict(X_tf) 
        
        #my_prediction = model.predict(data)
        
        if(my_prediction==1):
            val="Positive Review"
        else:
            val="Critical Review"
    

    return render_template('index.html', prediction_text=val,review=reviews)


if __name__ == "__main__":
    app.run(debug=True)
