import tweepy
import csv

# Below the twitter api key is input, Twitter TOS does not allow me to publish it, more information can be found here https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api
consumer_key = HIDDEN
consumer_secret = HIDDEN
access_token = HIDDEN
access_token_secret = HIDDEN

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

class CustomStreamListener(tweepy.StreamListener):
    def on_status(self, status):
        with open('tweets.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([status.created_at, status.user.screen_name, status.text])

    def on_error(self, status_code):
        if status_code == 420:
            return False

keywords_to_track = ['#bigmac', '#happymeal', '#McDonalds', 'McNuggets', 'McFlurry', 'Quarter Pounder', 'French Fries', 'McMuffin', 'Filet-O-Fish']

l = CustomStreamListener()

stream = tweepy.Stream(auth, l)
stream.filter(track=keywords_to_track)
