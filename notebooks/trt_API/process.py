#!/usr/bin/env python
# -*- coding: utf-8 -*-
#--------------------------------------------------------
# This is a Twitter Research Tools (TRT) file that can
# be called from TRT notebooks to process Tweets that
# were collected and queried using other tools
# provided as part of the TRT package.
#
# NOTE: This is still currently under development.
#
# TODO: enter licensing information here
#
# TODO: make to handle emoticons
#
#--------------------------------------------------------
import json as js
import trt_API.org_research as org
import glob
import sys
import pandas as pd
import numpy as np
import copy



'''
This function loads all of the tweet objects from a
directory that contains subdirectories containing the
actual tweets. It is intended for use with the results
of from one of the MapReduce mapper files after
querying the terms, cashtags, usernames or hashtags of
interest.
'''
def loadTweetObjects(input_dir):
  tweet_objects = []
  dirs = glob.glob(input_dir)
  cterr = 0
  ctln = 0
  for dr in dirs:
      files = glob.glob(dr+'/*')
      for f in files:
              fin = open(f,'r')
              for line in fin:
                  ctln += 1
                  try:
                      dat = js.loads(line)
                      tweet_objects.append(dat)
                  except:
                      cterr += 1
                      continue
              fin.close()
  return tweet_objects



'''
This function loads the tweets output from the
loadTweetObjects() function and extracts the desired
fields from the Tweet json.
#TODO: work on problems with encoding errors and make
to handle emoticons
'''
def convertTweetsToDataframe(tweet_objects, ENGLISH, encoding=True):
  fout = open('tmp.dat','w')
  ctke = 0
  for ln in tweet_objects:
      try:
          entities = ln['entities']
          user_mentions = entities['user_mentions']
      except KeyError:
          continue
      try:
          date = ln['created_at']
          date = date.split()
          #print (date)
          date = str(date[1])+' '+str(date[2])+' '+str(date[3])+' '+str(date[5])
          favorite_count = ln['favorite_count']
          user = ln['user']
          tweet_id = ln['id_str']
          if ENGLISH == True:
              lang = user['lang']
              if lang != 'en' and lang != 'en-gb':
                  continue
          followers = user['followers_count']
          username = user['screen_name']
          location = user['location']
          if ln['truncated'] == True:
              xtweet = ln['extended_tweet']
              txt = xtweet['full_text']
          else:
              txt = ln['text'].replace('\n','')
              try:
                otweet = ln['retweeted_status']
                if otweet['truncated'] == True:
                  otweet = otweet['extended_tweet']
                  otweet = otweet['full_text']
                else:
                  otweet = otweet['text']
              except:
                otweet = 'None'
          txt = txt.replace('\n','').replace('\r','')
          otweet = otweet.replace('\n','').replace('\r','')
          fout.write(date + '\t' + str(favorite_count)\
            +','+str(followers)+',"'+str(username)+'","'\
            +str(location)+'","'+str(txt)+'","'+str(tweet_id)+'","'+str(otweet)+'"\n')
      except UnicodeEncodeError:
          continue
      except KeyError:
          ctke += 1
      except:
        continue
  fout.close()
  if encoding == True:
    df = pd.read_csv('tmp.dat',error_bad_lines=False,encoding='utf-8',\
                    header=None,names=['date','followers',\
                    'username','location','tweet','id','original_tweet'],\
                     delimiter=',',index_col=False)
    print('Loaded utf-8 df.')
  else:
    df = pd.read_csv('tmp.dat',error_bad_lines=False,\
                header=None,names=['date','followers',\
                'username','location','tweet','original_tweet'],\
                 delimiter=',',index_col=False)
  #  print("Loaded df.")
  print("Initial size: " + str(df.shape[0]))
  df.drop_duplicates(inplace=True)
  print("Dropping duplicates...")
  print("Final size: " + str(df.shape[0]))
  df = df.dropna()
  return df



'''
This function identifies possible cashtags and returns
a new dataframe with them.
'''
def extractPossibleCashtags(df):
  tmp = pd.DataFrame()
  ct = 0
  for i in range (df.shape[0]):
      try:
          if " $" in df.tweet.iloc[i]:
              tmp.append(df.iloc[i,:])
              ct += 1
      except TypeError:
          continue
  print("Total potential Cashtags: " + str(ct))
  return tmp



'''
This function removes all 'noisy terms' as determined
by the user and passed to the function as a list.
'''
def removeNoisyTerms(df, noisy_terms):
  init = df.shape[0]
  for i in range(len(noisy_terms)):
    df.bad = df['tweet'].str.lower().str.contains(\
      noisy_terms[i].lower())
    df = df[df.bad == False]
  print ('Removed ' + str(init-df.shape[0]) + " noisy"\
        + " terms.")
  df = df.dropna()
  return df



'''
This function finds all 'terms of interest' as determined
by the user and passed to the function as a list.
'''
def findTerms(df, terms_of_interest):
  dfs = []
  for i in range(len(terms_of_interest)):
    df.good = df['tweet'].str.lower().str.contains(\
      terms_of_interest[i].lower())
    tdf = df[df.good == True]
    dfs.append(tdf)
  df = dfs[0]
  for i in range(1,len(dfs)):
    tdf = dfs[i]
    df = pd.concat([df,tdf],axis=0)
  print ('Found ' + str(df.shape[0]) + " terms of "\
        + "interest.")
  return df 


'''
This function replaces all the tweet words which contains 
#, https, RT etc with an empty string.
'''

def cleanText(text):
    spec_chars = ['_Ubiquitous']
    for char in spec_chars:
        text = text.str.replace(char, ' ')
    return text


'''
This function removes all duplicates and returns a new
dataframe without duplicates.
'''
def removeRetweets(df):
  tdf = copy.deepcopy(df)
  init = df.shape[0]
  df['RT'] = None
  df['RT'][df.tweet.astype(str).str[0:2] == 'RT'] = df.tweet.str.split(':',expand=True).iloc[:,0]
  tdf['rt'] = tdf['tweet'].str.split(':',\
   expand=True).iloc[:,0]
  tdf.drop_duplicates('tweet', keep='first',\
    inplace=True)
  tdf.rt.head()
  print("Removed " + str(init-tdf.shape[0]) + \
    " duplicates.")
  tdf = tdf.dropna()
  return tdf, df
