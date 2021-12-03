# pip install tweepy
# pip install pandas
import tweepy
from tweepy import OAuthHandler
import pandas as pd
import time
import csv
import tweepy
import ssl
import cred as cred


consumer_key = cred.consumer_key
consumer_secret = cred.consumer_secret
access_token = cred.access_token
access_token_secret = cred.access_token_secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

ssl._create_default_https_context = ssl._create_unverified_context
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# update these for the tweet you want to process replies to 'name' = the account username and you can find the tweet id within the tweet URL
reader = csv.reader(open('/Users/rajuy/Desktop/twitter_research_tools/notebooks/airdata_full/month_1.csv')) #update the path of the input file
count=0
replies=[]
f = open('replies_clean.csv', 'a')
writer = csv.writer(f)
for i, j in enumerate(reader):
    print(count)
    try:
        count+=1
        print(i, j)
        tweet_id = j[2]
        name = j[3]
        for tweet in tweepy.Cursor(api.search,q='to:'+name, result_type='recent', wait_on_rate_limit=True).items(10000):
            if hasattr(tweet, 'in_reply_to_status_id_str'):
                if (tweet.in_reply_to_status_id_str==tweet_id):
                    with open('replies_clean.csv', 'w') as f:
                        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
                        print(row)
                        writer.writerow(row)
                        replies.append(tweet)
                    
    except tweepy.TweepError as e:
        print(e)

#writes into CSV once the entire reply data for the tweets are fetched
with open('replies_clean.csv', 'a') as f:
    csv_writer = csv.DictWriter(f, fieldnames=('user', 'text'))
    csv_writer.writeheader()
    for tweet in replies:
        row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
        print(row)
        csv_writer.writerow(row)
        
        
            
        
        
            
                
        
        
            
