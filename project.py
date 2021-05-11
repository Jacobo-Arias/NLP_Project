import json
import tweepy

with open("credentials.json") as f:
    credentials = json.load(f)

API_key = credentials["API_key"]
API_secret_key = credentials["API_secret_key"]
Bearer_token = credentials["Bearer_token"]

auth = tweepy.OAuthHandler(API_key, API_secret_key)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)