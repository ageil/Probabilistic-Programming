import json

import numpy as np
from asterixdb.asterixdb import AsterixConnection

con = AsterixConnection(server="http://localhost", port=19002)

# get data froma asterixdb
news_rated = con.query(
    """
    USE FactMap;

    SELECT f.*
    FROM fuzzyurljoin f
    WHERE f.r.reviewRating.bestRating >= f.r.reviewRating.ratingValue
    AND f.r.reviewRating.ratingValue >= f.r.reviewRating.worstRating
    AND f.r.reviewRating.bestRating > f.r.reviewRating.worstRating
"""
).results
print("Num rated news:", len(news_rated))

reviews_rated = con.query(
    """
    USE FactMap;

    SELECT f.*
    FROM facturljoin f
    WHERE f.r.reviewRating.bestRating >= f.r.reviewRating.ratingValue
    AND f.r.reviewRating.ratingValue >= f.r.reviewRating.worstRating
    AND f.r.reviewRating.bestRating > f.r.reviewRating.worstRating
"""
).results
print("Num rated corrections:", len(reviews_rated))

# set up new asterixdb tables
q = """
    USE FactMap;

    DROP DATASET matchNews IF EXISTS;
    DROP DATASET matchCorrections IF EXISTS;
    DROP TYPE MatchType IF EXISTS;
    DROP TYPE ReviewType IF EXISTS;
    DROP TYPE PostType IF EXISTS;

    CREATE TYPE ReviewType as {
            uid: string
        };

    CREATE TYPE PostType as {
            id: string
        };

    CREATE TYPE MatchType as {
        r: ReviewType,
        p: PostType
    };

    CREATE DATASET matchNews(MatchType)
         PRIMARY KEY r.uid, p.id;
    CREATE DATASET matchCorrections(MatchType)
         PRIMARY KEY r.uid, p.id;
    """
response = con.query(q)


# compute isFake scores
def isFake(rating, worst, best):
    z = (rating - worst) / (best - worst)
    return z <= 0.5


# update news pairs with isFake scores
claim_scores = dict()
story_scores = dict()

for review in news_rated:
    uid = review["r"]["uid"]
    url = review["r"]["claimAuthor"]["claimURL"]
    worst = review["r"]["reviewRating"]["worstRating"]
    best = review["r"]["reviewRating"]["bestRating"]
    rating = review["r"]["reviewRating"]["ratingValue"]
    isfake = isFake(rating, worst, best)

    claim_scores[uid] = isfake
    if url not in story_scores:
        story_scores[url] = isfake
    else:
        story_scores[url] = isfake or story_scores[url]

    review["r"]["reviewRating"]["isFakeClaim"] = claim_scores[uid]
    review["r"]["reviewRating"]["isFakeStory"] = story_scores[url]

    q = """
    USE FactMap;
    INSERT INTO matchNews
    ({0});
    """.format(
        json.dumps(review)
    )
    con.query(q)

print("Fake claims:", np.mean(list(claim_scores.values())))
print("Fake stories:", np.mean(list(story_scores.values())))

# update correction pairs with isFake scores
review_scores = dict()

for review in reviews_rated:
    uid = review["r"]["uid"]
    url = review["r"]["reviewUrl"]
    worst = review["r"]["reviewRating"]["worstRating"]
    best = review["r"]["reviewRating"]["bestRating"]
    rating = review["r"]["reviewRating"]["ratingValue"]
    isfake = isFake(rating, worst, best)

    review_scores[uid] = isfake
    review["r"]["reviewRating"]["isFakeClaim"] = review_scores[uid]

    q = """
    USE FactMap;
    INSERT INTO matchCorrections
    ({0});
    """.format(
        json.dumps(review)
    )
    con.query(q)

print("Fake claims reviewed:", np.mean(list(review_scores.values())))

# dump data to json
q = """
USE FactMap;
SELECT m.*
FROM matchNews m
"""
matchNews = con.query(q).results

q = """
USE FactMap;
SELECT m.*
FROM matchCorrections m
"""
matchCorrections = con.query(q).results

with open("NewsPairs.json", "a") as f:
    for record in matchNews:
        f.write(json.dumps(record) + "\n")

with open("CorrectionPairs.json", "a") as f:
    for record in matchCorrections:
        f.write(json.dumps(record) + "\n")
