#!/usr/bin/env python
# coding: utf-8

# # FlipKart Phone Reviews Classification

# ##  Data Extraction

# In[1]:


#Importing required libraries
import requests   
from bs4 import BeautifulSoup as bs 
import re 
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import pandas as pd


# In[ ]:


poco_reviews1=[]
head_reviews1=[]
rating_reviews1=[]
for i in range(1,300):
    poco=[]
    ratings=[]
    head=[]
    url="https://www.flipkart.com/poco-x2-atlantis-blue-128-gb/product-reviews/itm36af4a9c20dd5?pid=MOBFZGJ6GQRXFZGT&lid=LSTMOBFZGJ6GQRXFZGTRXXE2U&marketplace=FLIPKART&page="+str(i)
    url="https://www.flipkart.com/poco-x2-phoenix-red-128-gb/product-reviews/itm399dc084bcc97?pid=MOBFZGJ6AXGFTJSC&lid=LSTMOBFZGJ6AXGFTJSC77SOJU&aid=overall&certifiedBuyer=false&sortOrder=NEGATIVE_FIRST&page="+str(i)
    response = requests.get(url)
    soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
    reviews = soup.findAll("div",attrs={"class","t-ZTKy"})# Extracting the content under specific tags 
    head_review=soup.findAll("p",attrs={"class","_2-N8zT"})
    rating=soup.findAll("div",attrs={"class","_1BLPMq"})
    
    for rr in head_review:
        hd_review=soup.findAll("p",attrs={"class","_2-N8zT"})
    
    
    for i in range(len(reviews)):
        poco.append(reviews[i].text)
    for i in range(len(head_review)):
        
        head.append(head_review[i].text)
    for i in range(len(rating)):
        
        ratings.append(rating[i].text)
    poco_reviews1=poco_reviews1+poco
    head_reviews1=head_reviews1+head
    rating_reviews1=rating_reviews1+ratings
#print(rating_reviews1)


# In[ ]:


poco_reviews2=[]
head_reviews2=[]
rating_reviews2=[]
for i in range(1,2000):
    poco=[]
    ratings=[]
    head=[]
    url="https://www.flipkart.com/poco-x2-phoenix-red-128-gb/product-reviews/itm399dc084bcc97?pid=MOBFZGJ6AXGFTJSC&lid=LSTMOBFZGJ6AXGFTJSC77SOJU&aid=overall&certifiedBuyer=false&sortOrder=POSITIVE_FIRST&page="+str(i)
    response = requests.get(url)
    soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
    reviews = soup.findAll("div",attrs={"class","t-ZTKy"})# Extracting the content under specific tags 
    head_review=soup.findAll("p",attrs={"class","_2-N8zT"})
    rating=soup.findAll("div",attrs={"class","_1BLPMq"})
    
    for rr in head_review:
        hd_review=soup.findAll("p",attrs={"class","_2-N8zT"})
    
    
    for i in range(len(reviews)):
        poco.append(reviews[i].text)
    for i in range(len(head_review)):
        
        head.append(head_review[i].text)
    for i in range(len(rating)):
        
        ratings.append(rating[i].text)
    poco_reviews2=poco_reviews2+poco
    head_reviews2=head_reviews2+head
    rating_reviews2=rating_reviews2+ratings
#print(rating_reviews1)


# In[ ]:


final_poco=[]
final_head=[]
final_rating=[]
for i in range(len(poco_reviews2)):
    final_poco.append(poco_reviews2[i])
    final_head.append(head_reviews2[i])
    final_rating.append(rating_reviews2[i])
for i in range(len(poco_reviews1)):
    final_poco.append(poco_reviews1[i])
    final_head.append(head_reviews1[i])
    final_rating.append(rating_reviews1[i])
import pandas as pd
data=pd.DataFrame()
data['Head Review']=final_head
data['Detailed Review']=final_poco
data['Rating']=final_rating
data.to_csv('Reviews.csv')


# In[2]:


data_review=pd.read_csv('Reviews.csv')
data_review.columns
data_review=data_review.drop(['Unnamed: 0'], axis=1)
data_review


# In[3]:


data_review['Rating'].value_counts()


# ## Sentiment Analyzer

# In[4]:


import nltk
nltk.download('vader_lexicon')


# In[5]:


import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
analyser=open('SentimentAnalyser.pkl','wb')
pickle.dump(sid,analyser)
analyser.close()
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
data_review['positive'] = data_review['Detailed Review'].apply(get_sentiment, k='positive')
data_review['negative'] = data_review['Detailed Review'].apply(get_sentiment, k='negative')
data_review['neu'] = data_review['Detailed Review'].apply(get_sentiment, k='neu')
data_review['compound'] = data_review['Detailed Review'].apply(get_sentiment, k='compound')


# ## Data Pre-Processing

# In[6]:


#if Rating> 3, set Rating = 1
#if Rating<=2, set Rating = 0
#if Rating == 3, remove the rows. 
import numpy as np
data_review['Rating'] = np.where(data_review['Rating']<=2, 0, data_review['Rating'])
data_review['Rating'] = np.where(data_review['Rating']>3, 1, data_review['Rating'])
data_review.drop(data_review[data_review['Rating']==3].index, inplace = True) 
data_review


# In[7]:


data_review['Rating'].value_counts()


# In[8]:


import re
def remove_htmltags(s):
    res = re.sub('<.*?>','',s)
    return res
data_review['Detailed Review']=data_review['Detailed Review'].apply(lambda w : remove_htmltags(w))
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
data_review['Detailed Review'] = data_review['Detailed Review'].apply(preprocess)
data_review['Head Review'] = data_review['Head Review'].apply(preprocess)


# ## Exploratory Data analysis

# ### Univariate Analysis of Review Heading

# In[9]:


word_count = data_review['Head Review'].str.split().apply(len).value_counts()
word_dict = dict(word_count)
word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(word_dict))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(word_dict.values()))

plt.ylabel('Number of reviews')
plt.title('Words for each heading of the review')
plt.xticks(ind, list(word_dict.keys()))
plt.show()


# In[10]:


positive_word_count = data_review[data_review['Rating']==1]['Head Review'].str.split().apply(len)
positive_word_count = positive_word_count.values

critical_word_count = data_review[data_review['Rating']==0]['Head Review'].str.split().apply(len)
critical_word_count = critical_word_count.values


# In[11]:


plt.boxplot([positive_word_count, critical_word_count])
plt.xticks([1,2],('Positive Reviews','Critical Reviews'))
plt.ylabel('Words in Heading')
plt.grid()
plt.show()


# In[12]:


import seaborn as sns
plt.figure(figsize=(10,3))
sns.distplot(positive_word_count, hist=False, label="Positive Reviews")
sns.distplot(critical_word_count, hist=False, label="Critical Reviews")
plt.legend()
plt.show()


# ### Univariate Analysis of Detailed Review

# In[13]:


word_count = data_review['Detailed Review'].str.split().apply(len).value_counts()
word_dict = dict(word_count)
word_dict = dict(sorted(word_dict.items(), key=lambda kv: kv[1]))


ind = np.arange(len(word_dict))
plt.figure(figsize=(20,5))
p1 = plt.bar(ind, list(word_dict.values()))

plt.ylabel('Number of reviews')
plt.xlabel('Number of words in each review')
plt.title('Words for each review')
plt.xticks(ind, list(word_dict.keys()))
plt.show()


# In[14]:


sns.distplot(word_count.values)
plt.title('Words for each review')
plt.xlabel('Number of words in each review')
plt.show()


# In[15]:


positive_word_count = data_review[data_review['Rating']==1]['Detailed Review'].str.split().apply(len)
positive_word_count = positive_word_count.values

critical_word_count = data_review[data_review['Rating']==0]['Detailed Review'].str.split().apply(len)
critical_word_count = critical_word_count.values


# In[16]:


plt.boxplot([positive_word_count, critical_word_count])
plt.xticks([1,2],('Positive Reviews','Critical Reviews'))
plt.ylabel('Words in Heading')
plt.grid()
plt.show()


# In[17]:


plt.figure(figsize=(10,3))
sns.distplot(positive_word_count, hist=False, label="Positive Reviews")
sns.distplot(critical_word_count, hist=False, label="Critical Reviews")
plt.legend()
plt.show()


# In[18]:


from wordcloud import WordCloud
comment_words=""
for val in data_review['Detailed Review']:   
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
wordcloud = WordCloud(width = 800, height = 800,background_color ='white',min_font_size = 10).generate(comment_words) 
plt.figure() 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title('Word Cloud Detailed Review Data')
plt.show() 


# ### Distribution of Reviews

# In[19]:


# this code is taken from 
# https://matplotlib.org/gallery/pie_and_polar_charts/pie_and_donut_labels.html#sphx-glr-gallery-pie-and-polar-charts-pie-and-donut-labels-py


y_value_counts = data_review['Rating'].value_counts()
print("Number of projects thar are approved for funding ", y_value_counts[1], ", (", (y_value_counts[1]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")
print("Number of projects thar are not approved for funding ", y_value_counts[0], ", (", (y_value_counts[0]/(y_value_counts[1]+y_value_counts[0]))*100,"%)")

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
recipe = ["Postive Reviews", "Critical Reviews"]

data = [y_value_counts[1], y_value_counts[0]]

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment, **kw)

ax.set_title("Number of positive and critical reviews")

plt.show()


# ## Train Test Split

# In[20]:


y=data_review['Rating']
X = data_review.drop(['Rating','Head Review'], axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y)
y_test.value_counts()


# In[21]:


#plot bar graphs of y_train and y_test
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
y_train.value_counts().plot(kind='bar')

plt.xlabel("Ratings", labelpad=14)
plt.ylabel("Count of Ratings")
plt.title("Train Distribution")


# In[22]:


y_test.value_counts().plot(kind='bar')
plt.xlabel("Ratings", labelpad=14)
plt.ylabel("Count of Ratings")
plt.title("Test Distribution")


# ## Featurization

# ### Featurization of text features

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


# In[25]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print("="*100)


vectorizer_tf = TfidfVectorizer(min_df=10,ngram_range=(1,4))
vectorizer_tf.fit(X_train['Detailed Review'].values) # fit has to happen only on train data

vectorizer=open('Vectorizer.pkl','wb')
pickle.dump(vectorizer_tf,vectorizer)
vectorizer.close()

# we use the fitted TFIDFVectorizer to convert the text to vector
X_train_detreview_tfidf = vectorizer_tf.transform(X_train['Detailed Review'].values)
X_test_detreview_tfidf = vectorizer_tf.transform(X_test['Detailed Review'].values)

print("After vectorizations")
print(X_train_detreview_tfidf.shape, y_train.shape)
print(X_test_detreview_tfidf.shape, y_test.shape)
print("="*100)


# In[28]:


#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)

#print("="*100)


#vectorizer_tf = TfidfVectorizer(min_df=10,ngram_range=(1,4))
#vectorizer_tf.fit(X_train['Head Review'].values) # fit has to happen only on train data

# we use the fitted TFIDFVectorizer to convert the text to vector
#X_train_headreview_tfidf = vectorizer_tf.transform(X_train['Head Review'].values)
#X_test_headreview_tfidf = vectorizer_tf.transform(X_test['Head Review'].values)

#print("After vectorizations")
#print(X_train_headreview_tfidf.shape, y_train.shape)
#print(X_test_headreview_tfidf.shape, y_test.shape)
#print("="*100)


# ### Featurization of numerical features

# In[26]:


from sklearn.preprocessing import Normalizer
print("="*50,"positive","="*50)
normalizer = Normalizer()
normalizer.fit(X_train['positive'].values.reshape(1,-1))

Norm_pos=open('Norm_pos.pkl','wb')
pickle.dump(normalizer,Norm_pos)
Norm_pos.close()

X_train_positive_norm = normalizer.transform(X_train['positive'].values.reshape(1,-1))
X_test_positive_norm = normalizer.transform(X_test['positive'].values.reshape(1,-1))

X_train_positive_norm = X_train_positive_norm.reshape(-1,1)
X_test_positive_norm = X_test_positive_norm.reshape(-1,1)


print("After vectorizations")
print(X_train_positive_norm.shape, y_train.shape)
print(X_test_positive_norm.shape, y_test.shape)

print("="*50,"negative","="*50)
normalizer = Normalizer()
normalizer.fit(X_train['negative'].values.reshape(1,-1))

Norm_neg=open('Norm_neg.pkl','wb')
pickle.dump(normalizer,Norm_neg)
Norm_neg.close()


X_train_negative_norm = normalizer.transform(X_train['negative'].values.reshape(1,-1))
X_test_negative_norm = normalizer.transform(X_test['negative'].values.reshape(1,-1))

X_train_negative_norm = X_train_negative_norm.reshape(-1,1)
X_test_negative_norm = X_test_negative_norm.reshape(-1,1)


print("After vectorizations")
print(X_train_negative_norm.shape, y_train.shape)
print(X_test_negative_norm.shape, y_test.shape)

print("="*50,"neu","="*50)
normalizer = Normalizer()
normalizer.fit(X_train['neu'].values.reshape(1,-1))

Norm_neu=open('Norm_neu.pkl','wb')
pickle.dump(normalizer,Norm_neu)
Norm_neu.close()

X_train_neu_norm = normalizer.transform(X_train['neu'].values.reshape(1,-1))
X_test_neu_norm = normalizer.transform(X_test['neu'].values.reshape(1,-1))

X_train_neu_norm = X_train_neu_norm.reshape(-1,1)
X_test_neu_norm = X_test_neu_norm.reshape(-1,1)


print("After vectorizations")
print(X_train_neu_norm.shape, y_train.shape)
print(X_test_neu_norm.shape, y_test.shape)

print("="*50,"compound","="*50)
normalizer = Normalizer()
normalizer.fit(X_train['compound'].values.reshape(1,-1))

Norm_com=open('Norm_com.pkl','wb')
pickle.dump(normalizer,Norm_com)
Norm_com.close()

X_train_compound_norm = normalizer.transform(X_train['compound'].values.reshape(1,-1))
X_test_compound_norm = normalizer.transform(X_test['compound'].values.reshape(1,-1))

X_train_compound_norm = X_train_compound_norm.reshape(-1,1)
X_test_compound_norm = X_test_compound_norm.reshape(-1,1)


print("After vectorizations")
print(X_train_compound_norm.shape, y_train.shape)
print(X_test_neu_norm.shape, y_test.shape)


# In[27]:


#Concatinating all the features
from scipy.sparse import hstack
#X_tr_tf = hstack((X_train_detreview_tfidf, X_train_headreview_tfidf,X_train_positive_norm,X_train_negative_norm,X_train_neu_norm,X_train_compound_norm)).tocsr()
#X_te_tf = hstack((X_test_detreview_tfidf, X_test_headreview_tfidf,X_test_positive_norm,X_test_negative_norm,X_test_neu_norm,X_test_compound_norm)).tocsr()

X_tr_tf = hstack((X_train_detreview_tfidf,X_train_positive_norm,X_train_negative_norm,X_train_neu_norm,X_train_compound_norm)).tocsr()
X_te_tf = hstack((X_test_detreview_tfidf,X_test_positive_norm,X_test_negative_norm,X_test_neu_norm,X_test_compound_norm)).tocsr()



print("Final Data matrix")
print(X_tr_tf.shape, y_train.shape)
print(X_te_tf.shape, y_test.shape)
print("="*125)


# ## Model Building

# In[28]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.tree import DecisionTreeClassifier
import math
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
classifier = GridSearchCV(LogisticRegression(), param_grid,cv=5,scoring='roc_auc',return_train_score=True)
classifier.fit(X_tr_tf, y_train)


# In[29]:


best_param=classifier.best_params_
print("Best Hyperparameter: ",best_param)
p_C=best_param['C']


# In[30]:


from sklearn.metrics import roc_curve, auc


DT = LogisticRegression(C=p_C)
DT.fit(X_tr_tf, y_train)
classifier=open('classifier.pkl','wb')
pickle.dump(DT,classifier)
classifier.close()

y_train_pred = DT.predict_proba(X_tr_tf)    
y_test_pred = DT.predict_proba(X_te_tf)

train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred[:,1])
test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred[:,1])

plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("AUC ROC Curve")
plt.grid()
plt.show()


# In[31]:


def find_best_threshold(thresholdl, fpr, tpr):
    t = thresholdl[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[32]:


print("="*50,"Confusion Matrix","="*50)
from sklearn.metrics import confusion_matrix
best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)
print("Train confusion matrix")
print(confusion_matrix(y_train, predict_with_best_t(y_train_pred[:,1], best_t)))
cnf_train=confusion_matrix(y_train, predict_with_best_t(y_train_pred[:,1], best_t))
print("Test confusion matrix")
print(confusion_matrix(y_test, predict_with_best_t(y_test_pred[:,1], best_t)))
cnf_test=confusion_matrix(y_test, predict_with_best_t(y_test_pred[:,1], best_t))


# In[33]:


#Computing AUC_Score with best parameter
AUC_Score_test_LOG=metrics.roc_auc_score(y_test,y_test_pred[:,1])
print('AUC_Score on test data: ',AUC_Score_test_LOG)
AUC_Score_train_LOG=metrics.roc_auc_score(y_train,y_train_pred[:,1])
print('AUC_Score on train data: ',AUC_Score_train_LOG)


# In[34]:


#Computing Accuracy Score
from sklearn.metrics import accuracy_score
print("Train Accuracy Score")
Accuracy_Score_Train_LOG=accuracy_score(y_train, predict_with_best_t(y_train_pred[:,1], best_t))
print(Accuracy_Score_Train_LOG)
print("Test Accuracy Score")
Accuracy_Score_Test_LOG=accuracy_score(y_test, predict_with_best_t(y_test_pred[:,1], best_t))
print(Accuracy_Score_Test_LOG)


# In[ ]:




