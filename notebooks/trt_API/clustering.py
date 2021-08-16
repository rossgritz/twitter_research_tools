#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------------
# This is a Twitter Research Tools (TRT) file that can
# be called from TRT notebooks to cluster Tweets that
# were collected and queried using other tools 
# provided as part of the TRT package. 
#
# NOTE: This is still currently under development.
#
# TODO: enter licensing information here
#
#--------------------------------------------------------
from sklearn.cluster import KMeans as kmeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import nltk
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.stem import WordNetLemmatizer 

import pandas as pd
import numpy as np



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
This function is included for vectorization of the terms 
using the count vectorizer
'''
def countVectorizer(df, n_FEATURES, NGRAM, stop_words):
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=5,\
                                        max_features=n_FEATURES,\
                                        stop_words=stop_words,\
                                        ngram_range=(1,NGRAM),\
                                        tokenizer=LemmaTokenizer())
    countv = count_vectorizer.fit_transform(df.tweet)
    count_feature_names = count_vectorizer.get_feature_names()
    return countv, count_feature_names



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
This function packages all of the kmeans processing into a 
single function so that the details can be avoided.
'''
def LDA(vectors, n_TOPICS):
    lda = LatentDirichletAllocation(n_TOPICS, random_state=0,\
                                    max_iter=100).fit(vectors)
    lda_embedding = lda.transform(vectors)
    lda_embedding = (lda_embedding - lda_embedding.mean(axis=0))\
                    /lda_embedding.std(axis=0)
    return lda, lda_embedding



'''
This function computes the inertia for a range of n to m topics. 
It takes as arguments the number vectorized data, number of iterations,
n and m. It returns a plot of the intertie over the range.
'''
def optimalClustersKMeans(vectors, iterations=5, n=3, m=20):
	inertias = np.zeros([iterations,m-n+1])
	for j in range(iterations):
		for i in range(m-n+1):
			model = kmeans(n_clusters=i+n, init='k-means++',\
				max_iter=100, n_init=1, verbose=False)
			model.fit(vectors)
			inertias[j,i] = model.inertia_
	mean_inertias = []
	for j in range(m-n+1):
		mean_inertias.append((sum(inertias[:,j]))/iterations)
	ks = [x for x in range(n,m+1)]
	plt.plot(ks, mean_inertias, '-o')
	plt.xlabel('Number of Clusters, k')
	plt.ylabel('Inertia')
	plt.xticks(ks)
	plt.show()




'''
This function prints the results of the kmeans clustering, 
including the top words and top tweets for each cluster. The
number of clusters, top words and top tweets can all be set.
'''
def printClusterResults(df, km, tfidf, tfidf_feature_names, \
                      n_TOP_WORDS, n_TOPICS, n_TOP_TWEETS=0):
  preds = km.predict(tfidf)
  count = 0
  for i in range(n_TOPICS):#, idxs in enumerate(top_idx.T): 
      print("Topic {}:".format(count+1))
      topic = km.cluster_centers_[i]
      print(" ".join([str(tfidf_feature_names[j])+'\n' \
          for j in topic.argsort()[:-n_TOP_WORDS - 1:-1]]))
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



'''
This function prints the results of the LDA topic modeling, 
including the top words and top tweets for each topic. The
number of topics, top words and top tweets can all be set.
'''
def printLDA(df, lda, lda_embedding, vector_feature_names, \
             n_TOPICS, n_TOP_WORDS, n_TOP_TWEETS=0):
    top_idx = np.argsort(lda_embedding,axis=0)[-n_TOP_TWEETS:]
    count = 0
    for i, idxs in enumerate(top_idx.T): 
        print("Topic {}:".format(i+1))
        topic = lda.components_[i]
        print(" ".join([str(vector_feature_names[i])+'\n' \
                        for i in topic.argsort()[:-n_TOP_WORDS - 1:-1]]))
        if n_TOP_TWEETS == 0:
            continue
        ct = 0
        idx_list = []
        for idx in idxs:
            ct += 1
            if idx not in idx_list:
            	print(str(ct)+') '+df.iloc[idx]['tweet']+"\n")
            	idx_list.append(idx)
        count += 1
        print('\n\n')



'''
This function takes the embeddings from clustering or topic
modeling and returns a tSNE visulaization. Perplexity can be
passed as an argument but the cluster/topic labels are fixed.
'''
def tSNE(kmeans_embedding,perplexity,n_TOPICS,\
                title='t-SNE visualization of Twitter topics/clusters'):
    tsne = TSNE(random_state=0,metric='jaccard',init='pca',\
                perplexity=perplexity) 
    tsne_embedding = tsne.fit_transform(kmeans_embedding)
    tsne_embedding = pd.DataFrame(tsne_embedding,columns=['x','y'])
    tsne_embedding['hue'] = kmeans_embedding.argmax(axis=1) 
    topics = [i+1 for i in range(n_TOPICS)]
    cmap = matplotlib.cm.get_cmap('nipy_spectral_r')
    colors = []
    for i in range(n_TOPICS):
        val = (float(i))/(n_TOPICS-1)
        colors.append(cmap(val))
    legend_list = []
    for i in range((n_TOPICS)):   
        color = colors[i]
        legend_list.append(mpatches.Ellipse((0, 0), 1, 1, fc=color))
    matplotlib.rc('font',family='monospace')
    plt.style.use('ggplot')
    fig, axs = plt.subplots(1,1, figsize=(30, 20), facecolor='w', \
                            edgecolor='k')
    fig.subplots_adjust(hspace = .1, wspace=0)
    axs.set_facecolor('white')
    count = 0
    legend = []
    data = tsne_embedding
    scatter = axs.scatter(data=data,x='x',y='y',s=42,c=data['hue'],\
                          cmap='nipy_spectral_r')
    plt.suptitle(title,\
                 **{'fontsize':'56','weight':'bold'},ha='center')
    fig.legend(legend_list,topics,loc=(0.05,0.75),ncol=3,fontsize=36)
    plt.subplots_adjust(top=0.85)
    plt.show()