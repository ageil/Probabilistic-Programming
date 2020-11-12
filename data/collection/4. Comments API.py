import numpy as np
import praw
from psaw import PushshiftAPI
from asterixdb.asterixdb import AsterixConnection
from credentials import CLIENT_ID, CLIENT_SECRET, PASSWORD, USERNAME
from datetime import datetime
import json


con = AsterixConnection(server='http://localhost', port=19002)


corrections = con.query('''
    USE FactMap;

    SELECT m.*
    FROM matchCorrections m;
    ''').results



news = con.query('''
    USE FactMap;
    
    SELECT m.*
    FROM matchNews m;
''').results



reddit = praw.Reddit(
    user_agent="Comment Extraction",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD
)
api = PushshiftAPI(reddit)


post_ids = {n['p']['id']: n['p']['created_utc'] for n in news+corrections}  # post_id: created_utc
failures = dict()

for idx, (pid, created_utc) in enumerate(post_ids.items()):
    if idx % 100 == 0:
        print('{}/{}'.format(idx, len(post_ids.keys())))
    try:
        post_datetime = datetime.strptime(created_utc, '%Y-%m-%dT%H:%M:%S.000Z')
        comments = list(api.search_comments(link_id=pid))
        num_comments = len(comments)
        comment_data = {'pid': pid, 'api_num_comments': num_comments}
        comment_data['comments'] = []
        for c in comments:
            delta_sec = datetime.fromtimestamp(c.created_utc) - post_datetime
            attributes = dict()
            attributes['cid'] = c.id
            attributes['author'] = c.author.name if c.author else None  # None if author name deleted(?)
            attributes['created_utc'] = c.created_utc
            attributes['ups'] = c.ups
            attributes['downs'] = c.downs
            attributes['body_len'] = len(c.body)
            attributes['parent_id'] = c.parent_id
            attributes['delta_seconds'] = delta_sec.seconds
            comment_data['comments'].append(attributes)
        with open('/Users/ageil/Github/FactMap/Data/comments.json', 'a') as f:
            json.dump(comment_data, f)
            f.write('\n')
    except:
        failures[pid] = created_utc

print('Failed:', len(failures.keys()))
with open('/Users/ageil/Github/FactMap/Data/comments_failed.json', 'a') as f:
    json.dump(failures, f)

