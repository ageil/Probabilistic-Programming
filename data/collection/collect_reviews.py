import json
import os
from datetime import datetime
from glob import glob

import requests


def getReviews(offset=0):
    params = {
        "hl": "",  # the language to search
        "num_results": 1000,
        "query": "list:recent",
        "force": "false",
        "offset": offset,
    }
    headers = {
        "dnt": "1",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "en-GB,en;q=0.9,it-IT;q=0.8,it;q=0.7,en-US;q=0.6",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36",
        "accept": "application/json, text/plain, */*",
        "referer": "https://toolbox.google.com/factcheck/explorer/search/list:"
        "recent;hl=en;gl=",
        "authority": "toolbox.google.com",
        "cookie": os.environ.get("GOOGLE_FACTCHECK_EXPLORER_COOKIE")
        # GOOGLE_FACTCHECK_EXPLORER_COOKIE = authorization cookie
    }
    response = requests.get(
        "https://toolbox.google.com/factcheck/api/search",
        params=params,
        headers=headers,
    )

    if response.status_code != 200:
        raise ValueError(response.status_code)

    content = json.loads(response.text[5:])[0][1:3]
    today = datetime.now().strftime("%B %d, %Y")
    path = "/Users/ageil/Github/FactMap/Data/claimreviews/raw/" + today + "/"
    filename = "raw_{0}.json".format(offset)

    if not os.path.isdir(path):
        os.makedirs(path)
    with open(path + filename, "w") as f:
        json.dump(content, f, indent=2)

    return content


def collect():
    offset = 0
    claims = []
    tags = []

    while True:
        print("offset", offset)
        content = getReviews(offset=offset)
        if not content[0]:
            break
        offset += len(content[0])
        claims.append(content[0])
        tags.append(content[1])

    return claims, tags


def parseReviews(reviews):
    results = []
    for idx, r in enumerate(reviews):
        has_rtng = len(r[0][3][0]) > 9 and r[0][3][0][9] and len(r[0][3][0][9])
        try:
            claimReview = {
                "reviewUrl": r[0][3][0][1],
                "claimReviewed": r[0][0],
                "lang": r[0][3][0][6],
                "countries": r[0][3][0][7],
                "claimReviewed_en": r[0][11] if len(r[0]) > 11 else None,
                "claimDate": r[0][2] if len(r[0]) > 2 else None,
                "reviewDate": r[0][3][0][2] if len(r[0][3][0]) > 2 else None,
                "reviewAuthor": {
                    "name": r[0][3][0][0][0],  # review author
                    "authorURL": r[0][3][0][0][1],
                },
                "reviewRating": {
                    "ratingValue": r[0][3][0][9][0] if has_rtng else -1,
                    "worstRating": r[0][3][0][9][1] if has_rtng else -1,
                    "bestRating": r[0][3][0][9][2] if has_rtng else -1,
                    "alternateName": r[0][3][0][3],
                },
                "claimAuthor": {
                    "name": r[0][1][0],  # claim author
                    "claimURL": r[0][4][0][1] if len(r[0][4]) else None,
                }
                if len(r[0][1])
                else {},
                "tagsRaw": [
                    {"keyword": tag[0], "probability": tag[1]}
                    for tag in r[0][8]
                    if len(tag) == 3
                ],
                "tagsNamed": [
                    {"keyword": raw_tags[tag[0]], "probability": tag[1]}
                    for tag in r[0][8]
                    if (tag[0] in raw_tags) and (len(tag) == 3)
                ],
                "reviewTitle": r[0][3][0][8],
            }
            results.append(claimReview)
        except IndexError as e:
            print(idx)
            print(json.dumps(r))
            raise (e)
    return results


# collect claimreviews
raw = collect()

# Load collected raw data:
path = "/Users/ageil/Github/FactMap/Data/claimreviews/raw/October 19, 2020/"
raw_paths = glob(path + "*.json")

raw = []
for path in raw_paths:
    with open(path) as f:
        raw.append(json.load(f))

raw_claims = [c for batch in raw for c in batch[0]]
raw_tags = {c[0]: c[1] for batch in raw for c in batch[1]}

claims = parseReviews(raw_claims)
path = "/Users/ageil/Github/FactMap/Data/claimreviews/claims2020.json"

with open(path, "w") as f:
    f.write("\n".join(json.dumps(c) for c in claims))
