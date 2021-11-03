#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------------
# This is a Twitter Research Tools (TRT) file that can
# be called from TRT notebooks. It contains specific 
# packages for sentiment analysis. 
#
# NOTE: This is under development. Currently only a 
# sentiment dictionary is available.
#
# TODO: enter licensing information here
# 
#--------------------------------------------------------
import matplotlib.pyplot as plt
from copy import deepcopy



'''
This function loads the sentiment dictionary.
'''
def getDictionary(dpath):
	fdict = open(dpath,'r')
	sent_dict = {}
	for line in fdict:
		line = line.strip('\n')
		line = line.split('\t')
		sent_dict[line[0]] = int(line[1])
	return sent_dict



'''
This function is for normalizing the sentiment and it gets the counts
for each of the hashtags/usernames.
'''
def getCounts(tweets, hashtags):
	for i, tag in enumerate(hashtags): hashtags[i] = tag.lower() 
	counts = [0 for i in range(len(hashtags))]
	for tweet in tweets:
		tweet = tweet.lower()
		for i, tag in enumerate(hashtags):
			if tag in tweet:
				counts[i] += 1
	tcounts = deepcopy(counts)
	return tcounts



'''
This function plots the total sentiment via a histogram with all bins
and via a positive negative histogram.
'''
def plotTotalSentiment(tweets, srange=(-8,8), dpath='../resources/AFINN-111.txt'):
	sent_dict = getDictionary(dpath)
	tweet_sent = []
	for tweet in tweets:
		tweet = tweet.lower()
		tweet = tweet.split(' ')
		sent = 0
		for word in tweet:
			word = word.strip('#')
			try:
				sent += sent_dict[word]
			except:
				pass
		tweet_sent.append(sent)
	# plot histogram
	bins = [s+0.5 for s in range(srange[0],srange[1])] 
	plt.hist(tweet_sent,bins=bins)
	plt.xlim(srange[0],srange[1])
	plt.title('SENTIMENT HISTOGRAM')
	plt.show()
	# plot negative positive only
	sign = []
	for sent in tweet_sent:
		if sent > 0:
			sign.append(1)
		elif sent < 0:
			sign.append(-1)
		else:
			pass
	plt.hist(sign,bins=[-10,0,10],rwidth=0.9)
	plt.xlim(-10,10) 
	plt.title('NEGATIVE                            POSITIVE')
	plt.show()



'''
This plots sentiment by hashtag or username and takes a list of hastags/usernames
as an argument. It takes a variable normalized as an argument which determines 
whether the output is normalized.
'''
def computeHashtagSentiment(tweets, hashtags, normalized=True, plot=True, \
							dpath='../resources/AFINN-111.txt'):
	sent_dict = getDictionary(dpath)
	tcounts = getCounts(tweets, hashtags)
	for i, tag in enumerate(hashtags): hashtags[i] = tag.lower() 
	counts = [0 for i in range(len(hashtags))]
	tag_sents = {}
	tag_sent_list = {}
	tag_sent_values = {}
	for tag in hashtags:
		tag = tag.lower()
		tag_sents[tag] = 0
		tag_sent_list[tag] = []
		tag_sent_values[tag] = 0
	for tweet in tweets:
		tweet = tweet.lower()
		tweet = tweet.split(' ')
		sent = 0
		for word in tweet:
			word = word.strip('#')
			try:
				sent += sent_dict[word]
			except:
				pass
		for i, tag in enumerate(hashtags):
			tag = tag.lower()
			if tag in tweet:
				try:
					tag_sents[tag] += sent
					tag_sent_list[tag].append(sent)
				except:
					pass
	tag_sents_ = deepcopy(tag_sents)
	if normalized:
		for i, key in enumerate(tag_sents_.keys()):
			if tag_sents_[key] == 0:
				continue
			tag_sents_[key] = tag_sents_[key]/tcounts[i]
	if plot:
		plt.bar(tag_sents_.keys(),tag_sents_.values(),color=['r','g','b','k'])
		plt.xticks(rotation='vertical')
		plt.show()
	return tag_sents_



'''
This computes sentiment within the dataframe so that users may sort Tweets in
other ways and have the sentiment scores for plotting and comparison.
'''
def computeDataframeSentiment(df, dpath='../resources/AFINN-111.txt'):
	df['sentiment'] = 0
	sent_dict = getDictionary(dpath)
	for i in range(df.shape[0]):
		tweet = df.tweet.iloc[i]
		tweet = tweet.lower()
		tweet = tweet.split(' ')
		sent = 0
		for word in tweet:
			word = word.strip('#')
			try:
				sent += sent_dict[word]
			except:
				pass
		df.sentiment.iloc[i] += sent
	return df



'''
This function takes dataframes and compares them visually using a bar chart. 
It requires a list of dataframes as an argument, and can be passed a list of
names for each dataframe that must be equal in length to the list of dataframes.
'''
def compareSentimentByDataframe(dfs=[],normalized=True,names=[],plot=True,\
								dpath='../resources/AFINN-111.txt'):
	sents = []
	sent_dict = getDictionary(dpath)
	for i in range(len(dfs)):
		df = dfs[i]
		tweets = df.tweet
		tweet_sent = []
		ct = 1
		for tweet in tweets:
			tweet = tweet.lower()
			tweet = tweet.split(' ')
			sent = 0
			for word in tweet:
				word = word.strip('#')
				try:
					sent += sent_dict[word]
				except:
					pass
			ct += 1
			tweet_sent.append(sent)
		if normalized:
			sents.append(sum(tweet_sent)/ct)
		else:
			sents.append(sum(tweet_sent))
	if len(names) == 0:
		names = ['Topic '+str(x+1) for x in range(len(sents))]
	elif len(names) != len(sents):
		print("ERROR: The number of names must match the number of dataframes!!")
		return 0
	if plot:
		plt.bar(names,sents,color=['r','g','b','k','m'])
		plt.xticks(rotation='vertical')
		plt.show()

