
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------------
# This is a Twitter Research Tools (TRT) file that can
# be called from TRT notebooks. It contains specific 
# packages for basic descriptive analysis. 
#
# NOTE: This is under development.
#
# TODO: enter licensing information here
# 
#--------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd



'''
This counts hastags or usernames and plots or prints the results
based on flags passed as arguments to the function.
'''
def countItems(hashtags=[],tweets=[],PLOT=False,PRINT=True):
	for i, tag in enumerate(hashtags): 
		hashtags[i] = tag.lower()
	counts = [0 for i in range(len(hashtags))]
	for tweet in tweets:
		tweet = tweet.lower()
		for i, tag in enumerate(hashtags):
			if tag in tweet:
				counts[i] += 1
	if PRINT:
		for i, tag in enumerate(hashtags):
			print(tag.upper()+':   '+str(counts[i]))
	if PLOT:
		plt.bar(hashtags,counts,color='green')
		plt.xlabel("Hashtags")
		plt.ylabel("Frequency")
		plt.title("Hashtag Frequencies")
		plt.xticks(rotation=90)
		plt.show()
	return counts



'''
This function first converts the raw dates extracted from the Tweet
jsons into datetime objects. It then sorts the dataframe by date. If
the PRINT_TOP flag is set to true it will print the top n (TOP) Tweets
by date.
'''
def sortByDate(df,SORTED=True,PRINT_TOP=False,TOP=10):
	if not SORTED:
		date = df.date
		date = date.str.split('\t')
		date = date.apply(pd.Series)
		df['date'] = date.iloc[:,0]
		df['date'] = pd.to_datetime(df['date'])
	df.sort_values(by='date',ascending=True,inplace=True)
	if PRINT_TOP:
		print(df.head(TOP))
	return df, True



'''
This function takes as input a list of tweets (tweets) and returns a 
plot of the top hashtags sorted by frequency. The number of top hashtags
to view is set by the argument n.
'''
def topHashtags(tweets,n):
	topHashtags = {}
	for tweet in tweets:
		tweet = tweet.lower()
		tweet = tweet.split(' ')
		for word in tweet:
		 	if len(word) == 0:
		 		continue
		 	if word[0] == '#':
		 		try:
		 			topHashtags[word.strip('!()-[]{};:\'"\\,<>./?@#$%^&*_~')] += 1
		 		except:
		 			topHashtags[word.strip('!()-[]{};:\'"\\,<>./?@#$%^&*_~')] = 1
	sorted_by_value = sorted(topHashtags.items(), key=lambda kv: kv[1])
	d = dict(sorted_by_value)
	dkeys = [d.keys()]
	dkeys = list(dkeys[0])
	dvals = [d.values()]
	dvals = list(dvals[0])
	print (dvals[-n:])
	plt.bar(dkeys[-n:],dvals[-n:],color=['g','m','k'])
	plt.xticks(rotation='vertical')
	plt.show()



'''
This function takes as input a list of tweets (tweets) and returns a 
plot of the top usernames mentioned sorted by frequency. The number of 
top usernames to view is set by the argument n.
'''
def topUserMentions(tweets,n):
	topHashtags = {}
	for tweet in tweets:
		tweet = tweet.lower()
		tweet = tweet.split(' ')
		for word in tweet:
		 	if len(word) == 0:
		 		continue
		 	if word[0] == '@':
		 		try:
		 			topHashtags[word.strip('!()-[]{};:\'"\\,<>./?@#$%^&*_~')] += 1
		 		except:
		 			topHashtags[word.strip('!()-[]{};:\'"\\,<>./?@#$%^&*_~')] = 1
	sorted_by_value = sorted(topHashtags.items(), key=lambda kv: kv[1])
	d = dict(sorted_by_value)
	dkeys = [d.keys()]
	dkeys = list(dkeys[0])
	dvals = [d.values()]
	dvals = list(dvals[0])
	print (dvals[-n:])
	plt.bar(dkeys[-n:],dvals[-n:],color=['g','m','k'])
	plt.xticks(rotation='vertical')
	plt.show()

