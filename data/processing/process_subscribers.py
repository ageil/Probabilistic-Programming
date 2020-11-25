from datetime import datetime
from collections import defaultdict
import json

# load matches
news = []
with open('../results/NewsPairs.json') as f:
    for line in f:
        news.append(json.loads(line))

corrections = []
with open('../results/CorrectionPairs.json') as f:
    for line in f:
        corrections.append(json.loads(line))

# load historical subscriber counts scraped from subredditstats.com
subscribers = {}
with open('/Users/ageil/Desktop/subscribers.json') as f:
    for line in f:
        data = json.loads(line)
        subreddit = data['subreddit']
        history = []
        for h in data['history']:
            date = datetime.strptime(h['date'], '%Y-%m-%d')
            count = h['subscriber_count']
            history.append({'date': date, 'count': count})        
        subscribers[subreddit] = history

# load failed subs
with open('../results/failed.txt', 'r') as f:
    failed_subs = f.readlines()
    failed_subs = [sub.split('\n')[0] for sub in failed_subs]

# manually obtain missing subscriber counts from subredditstats.com
manual_dates = [('PoliticalOasis', '2018-10-11T01:40:30.000Z', 9),
 ('usa', '2018-03-15T18:09:13.000Z', 9702),
 ('usa', '2018-03-15T18:09:13.000Z', 9702),
 ('usa', '2018-01-13T22:25:18.000Z', 9337),
 ('usa', '2018-07-19T04:13:00.000Z', 10539),
 ('usa', '2019-01-31T16:50:07.000Z', 9337),
 ('theItaly', '2018-08-08T08:22:08.000Z', 104),
 ('usa', '2018-03-04T05:33:30.000Z', 9631),
 ('usa', '2016-08-08T17:58:23.000Z', 10761),
 ('usa', '2018-07-17T23:56:27.000Z', 10539),
 ('theItaly', '2018-07-18T08:33:44.000Z', 57),
 ('usa', '2017-12-18T16:10:11.000Z', 9285),
 ('usa', '2018-05-05T02:22:28.000Z', 10200),
 ('usa', '2017-12-12T17:39:20.000Z', 9267),
 ('theItaly', '2018-08-08T08:22:08.000Z', 104),
 ('usa', '2018-03-04T05:33:30.000Z', 9631),
 ('usa', '2018-05-05T02:22:28.000Z', 10200),
 ('usa', '2017-12-12T17:39:20.000Z', 9267),
 ('theItaly', '2018-07-18T08:33:44.000Z', 57),
 ('usa', '2017-12-18T16:10:11.000Z', 9285),
 ('usa', '2018-07-17T23:56:27.000Z', 10539),
 ('theItaly', '2018-07-18T08:33:44.000Z', 57),
 ('usa', '2018-05-05T02:22:28.000Z', 10200),
 ('usa', '2018-07-17T23:56:27.000Z', 10539),
 ('theItaly', '2018-08-08T08:22:08.000Z', 104)]

fixed_dates = defaultdict(list)
for (sub, date, count) in manual_dates:
    fixed_dates[sub] += [(datetime.strptime(date, '%Y-%m-%dT%H:%M:%S.000Z'), count)]

# clean data from non-subreddits (user groups)
clean_news = []
for n in news:
    if n['p']['subreddit'] in ['usa', 'PoliticalOasis', 'theItaly']:
        clean_news.append(n)
    if n['p']['subreddit'] not in failed_subs:  # i.e. a user page (not a subreddit)
        clean_news.append(n)

clean_corrections = []
for n in corrections:
    if n['p']['subreddit'] in ['usa', 'PoliticalOasis', 'theItaly']:
        clean_corrections.append(n)
    if n['p']['subreddit'] not in failed_subs:
        clean_corrections.append(n)

# merge subscriber counts by subreddit/date
clean_news2 = []
for n in clean_news:
    review = n['r']
    p = n['p']
    
    subreddit = n['p']['subreddit']
    postdate = datetime.strptime(n['p']['created_utc'], '%Y-%m-%dT%H:%M:%S.000Z')
    
    count = 0
    if subreddit in subscribers.keys():
        for x in subscribers[subreddit]:
            if x['date'] <= postdate:
                count = x['count']
            else:
                break
    else:
        for d, c in fixed_dates[subreddit]:
            if d <= postdate:
                count = c
            else:
                break    
    p['subscribers'] = count
    clean_news2.append({'r': review, 'p': p})

clean_corrections2 = []
for n in clean_corrections:
    review = n['r']
    p = n['p']
    
    subreddit = n['p']['subreddit']
    postdate = datetime.strptime(n['p']['created_utc'], '%Y-%m-%dT%H:%M:%S.000Z')
    
    count = 0
    if subreddit in subscribers.keys():
        for x in subscribers[subreddit]:
            if x['date'] <= postdate:
                count = x['count']
            else:
                break
    else:
        for d, c in fixed_dates[subreddit]:
            if d <= postdate:
                count = c
            else:
                break    
    p['subscribers'] = count
    clean_corrections2.append({'r': review, 'p': p})

# output merged data to file
for n in clean_news2:
	with open("/Users/ageil/Desktop/news_clean.json", "a") as f:
	    json.dump(n, f)
	    f.write("\n")

for n in clean_corrections2:
	with open("/Users/ageil/Desktop/corrections_clean.json", "a") as f:
	    json.dump(n, f)
	    f.write("\n")
