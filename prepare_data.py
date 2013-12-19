__author__ = 'eddiexie'
import re

import pymongo
import numpy as np
import scipy

from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def check_user(tweets):
    weekend = 0
    weekday = 0
    for tweet in tweets:
        parsed_time = datetime.fromtimestamp(float(tweet['created_time']))
        if parsed_time.day == 6 or parsed_time.day == 0:
            weekend += 1
        else:
            weekday += 1
        if weekend >= 1  and weekday >= 1:
            return True
    return False

def parse_data(tweets):
    user_ids = {}   # user given id
    u_index = 0
    for n, tweet in enumerate(tweets):
        if tweet['user']['id'] not in user_ids:
            user_ids[tweet['user']['id']] = u_index
            u_index += 1

    n_doc = len(tweets)
    n_user = len(user_ids)

    texts = [tweet['text'] for tweet in tweets]

    vectorizer = CountVectorizer(min_df = 3, max_df = 0.5, ngram_range=(1,1), stop_words='english' )
    doc = vectorizer.fit_transform(texts)

    #doc = scipy.sparse.dok_matrix(doc)

    user_docs = [[] for x in range(n_user)]

    day_of_week = []
    created_time = np.zeros((n_doc, 1))
    lat = np.zeros((n_doc, 1))
    lng = np.zeros((n_doc, 1))

    doc_user = []
    user_day_count = np.zeros((n_user, 2))
    user_loc = [[] for x in xrange(n_user) ]
    doc_loc = []

    user_real_ids = {}


    for n, tweet in enumerate(tweets):
        user_id = tweet['user']['id']
        parsed_time = datetime.fromtimestamp(float(tweet['created_time']))
        created_time[n] = parsed_time.hour + parsed_time.minute*1.0/60
        if parsed_time.day == 6 or parsed_time.day == 0:
            day_of_week.append(1)
            user_day_count[user_ids[user_id], 1] += 1
        else:
            day_of_week.append(0)
            user_day_count[user_ids[user_id], 0] += 1
        lat[n] = tweet['location']['latitude']
        lng[n] = tweet['location']['longitude']
        user_docs[user_ids[user_id]].append(n)
        doc_user.append(user_ids[user_id])

        user_loc[user_ids[user_id]].append( (lat[n], lng[n]) )
        doc_loc.append([lat[n], lng[n]])


    for id in user_ids.keys():
        user_real_ids[user_ids[id]] = id


    for u in range(n_user):
        n_row = len(user_loc[u])
        n_col = 2
        user_loc[u] = np.asarray(user_loc[u]).reshape((n_row, n_col))

    doc_loc = np.asarray(doc_loc).reshape((n_doc, 2))
    print doc_loc.shape
    print user_ids
    return doc_user, user_real_ids, user_docs, user_day_count, user_loc, doc_loc, created_time, day_of_week, lat, lng, \
           texts, doc, vectorizer


def get_data(max_users = 30):
    """Get data from mongodb"""

    #cache here


    mongo_db = pymongo.Connection('grande.rutgers.edu', 27017)['citybeat_production']
    tweets_collection = mongo_db['tweets']


    test_tweets = []
    seed_users = []



    try:
        with open('./cache_tweets.pkl'):
            tweets, test_tweets = pickle.load(open('./cache_tweets.pkl'))
    except:
        print 'in'
        # not here. fetch
        tweets = []
        for n, tweet in enumerate(tweets_collection.find({"created_time": {"$gte":"1380643200", "$lt":"1380902400"}})):
            tweet['text'] = re.sub(r"(?:\@|https?\://)\S+", "", tweet['text'])
            tweet['text'] = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet['text'], flags=re.MULTILINE)
            tweets.append(tweet)
            print n

        #print 'len of tweets ', len(tweets), 'len of test = ', len(test_tweets)
        test_tweets = tweets[-100:-1]
        #pickle.dump((tweets, test_tweets), open('./cache_tweets.pkl','w'))

    tweets = [tweet for tweet in tweets if len(tweet['text'].split(' ')) >= 10]






    return tweets, test_tweets


def get_formatted_data(max_user):
    tweets, test_tweets = get_data(max_user)
    return (parse_data(tweets), parse_data(test_tweets))

#get_data()