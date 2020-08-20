#--------------------------------------------------------
# This is a Twitter Research Tools (TRT) file that can
# be called from TRT notebooks to cluster Tweets that
# were collected and queried using other tools 
# provided as part of the TRT package. 
#
# TODO: enter licensing information here
#
#--------------------------------------------------------
from sklearn.cluster import KMeans as kmeans
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import WordNetLemmatizer 



'''
This Class implements lemmatization to remove the unecessary
parts of the word and reduce words to their roots. It then
tokenizes words for processing in clustering.
'''
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) \
        for t in tokenizer.tokenize(doc)]



'''
This function is included for vectorization of the terms 
using the term-frequency inverse document frequency 
technique.
'''
def tfidf(df, n_FEATURES, NGRAM, stop_words):
  tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=5,\
    max_features=n_FEATURES,stop_words=stop_words,\
    ngram_range=(1,NGRAM),tokenizer=LemmaTokenizer())
  tfidf = tfidf_vectorizer.fit_transform(df.tweet)  
  tfidf_feature_names = tfidf_vectorizer.get_feature_names()
  return tfidf, tfidf_feature_names



'''
This function packages all of the kmeans processing into a 
single function so that the details can be avoided.
'''
def KMeans(tfidf, n_TOPICS):
  km = kmeans(n_clusters=n_TOPICS, init='k-means++',\
              max_iter=100, n_init=1, verbose=True)
  km.fit(tfidf)
  kmeans_embedding = km.transform(tfidf)
  kmeans_embedding = -(kmeans_embedding -\
                  kmeans_embedding.mean(axis=0))\
                  /kmeans_embedding.std(axis=0)
  return km, kmeans_embedding



'''
This function prints the results of the kmeans clustering, 
including the top words and top tweets for each cluster. The
number of clusters, top words and top tweets can all be set.
'''
def printClusterResults(df, km, tfidf, tfidf_feature_names, \
                      n_TOP_WORDS, n_TOP_TWEETS, n_TOPICS):
  preds = km.predict(tfidf)
  count = 0
  for i in range(n_TOPICS):#, idxs in enumerate(top_idx.T): 
      print("Topic {}:".format(count))
      topic = km.cluster_centers_[i]
      print(" ".join([str(tfidf_feature_names[i])+'\n' \
          for i in topic.argsort()[:-n_TOP_WORDS - 1:-1]]))
      ct = 0
      top_tweets = []
      t = 0
      for t in range(len(preds)):
          if preds[t] == i:
              top_tweet = df.iloc[t]['tweet']
              if top_tweet not in top_tweets:
                  print(str(ct+1) + ") " + top_tweet)
                  top_tweets.append(top_tweet)
                  ct += 1
          if ct == n_TOP_TWEETS:
              break
          t += 1
      count += 1
      print('\n\n')