import json
import tweepy
 
import pandas as pd
import csv
import re 
import string
# import preprocessor as p

with open("credentials.json") as f:
    credentials = json.load(f)

consumer_key = credentials["API_key"]
consumer_secret = credentials["API_secret_key"]

access_token= credentials["Access_token"]
access_token_secret = credentials["Access_secret_token"]
# Bearer_token = credentials["Bearer_token"]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print(tweet.text)

csvFile = open('tweets.csv', 'a')
csvWriter = csv.writer(csvFile)

search_words = "claro OR tigo internet @Tigo_Colombia OR @ClaroColombia"      # enter your words
new_search = search_words + " -filter:retweets"
tweets_extraidos = tweepy.Cursor(api.search,q=new_search,
                           lang="es",
                           since_id=0)

tweets_extraidos_items = tweets_extraidos.items()

for tweet in tweets_extraidos_items:
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8'),tweet.user.screen_name.encode('utf-8'), tweet.user.location.encode('utf-8')])
