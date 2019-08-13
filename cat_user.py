"""
This script allows us to either pass a reddit username and categorize the user
as either naughty or nice. We can also run the sript so that anyone can activate
by calling the activation_phrase in a reddit comment.
"""
import classifier_mod as c
import praw
from collections import Counter
import argparse

# created an argeparser so that one can use the demo mode or normal return
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True,
	help="demo or run")
args = vars(ap.parse_args())

# this method looks through the past 50 comments a user has, classifies it, and determines
# whether the user has been naughty or nice depending on which is counted more
def naughty_or_nice(user, post=None):
    user_comments = []

    # checks the 50 most recent comments and adds to list for analyzing
    comments = reddit.redditor(user).comments.new(limit=10)
    for comment in comments:
        sent, confidence = c.sentiment(comment.body)
        if sent == "naughty" and confidence == 1.0:  # only adds a naughty comment if it's 100% sure it's negative
            user_comments.append(sent)

        elif sent == "nice":
            user_comments.append(sent)

    user_sentiment = Counter(user_comments)
    # if more naughty comments, reply that user has been naughty
    if user_sentiment["naughty"] >= user_sentiment["nice"]:
        if post != None:
            post.reply("Ho ho ho, {} is getting a lump of coal. Try being nicer next year, ho ho ho.".format(user))
    # if more nice comments, reply that user has been nice
    else:
        if post != None:
            post.reply("Ho ho ho, {} has been a nice user this year! Come to the North Pole to pick up your gift".format(user))

# authenticating user (Is there a better way to do this without hardcoding my id/password)
reddit = praw.Reddit(client_id='client_id',
                     client_secret='client_secret',
                     password='password',
                     user_agent='testscript_by_username',
                     username='username')

activation_phrase = "!Santa"
active = True

# if demo chosen in terminal, let's user enter a name
if args["mode"] == "demo":
    name = input("Ho ho ho, who's name are we looking up?\t")
    naughty_or_nice(name)

# if run chosen, the bot waits for the activation term so that it can implement the method
elif args["mode"] == "run":
    print("Waiting...")
    while active:
        comment_stream = reddit.subreddit("all").comments(limit = None)
        for comment in comment_stream:
            if activation_phrase.encode('utf-8') in comment.body.encode('utf-8'):
                naughty_or_nice(comment.parent().author.name, comment)
