import json

import numpy as np
import torch


def load_comments():
    comments = dict()
    with open("../data/results/Comments.json") as f:
        for line in f:
            post = json.loads(line)
            comments[post["pid"]] = post["api_num_comments"], post["comments"]
    return comments


def load_corrections():
    corrections = []
    with open("../data/results/CorrectionPairs.json") as f:
        for line in f:
            corrections.append(json.loads(line))
    return corrections


def load_news():
    news = []
    with open("../data/results/NewsPairs.json") as f:
        for line in f:
            news.append(json.loads(line))
    return news


def load_raw_data():
    comments = load_comments()
    corrections = load_corrections()
    news = load_news()
    return comments, corrections, news


def processData(items, comments, comments_only=False, minutes=60):
    # story level
    sid_set = {n["r"]["uid"] for n in items}
    sids = list(sorted(sid_set))
    sid_indices = {sid: i for i, sid in enumerate(sids)}

    country_set = {c for n in items for c in n["r"]["countries"]}
    countries = list(sorted(country_set))
    country_indices = {country: i for i, country in enumerate(countries)}

    author_set = {n["r"]["reviewAuthor"]["authorURL"] for n in items}
    authors = list(sorted(author_set))
    author_indices = {author: i for i, author in enumerate(authors)}

    story_claim_titles = [""] * len(sids)

    s_data = np.zeros((len(sids), 1 + len(countries) + len(authors)))
    s_data[:, 0] = 1

    # subreddit level
    num_bins = 25
    counts = np.array([max(0, n["p"]["subscribers"]) for n in items])
    subreddit_bins = np.quantile(counts, np.linspace(0, 1, num_bins + 1))
    subreddit_bins[-1] = subreddit_bins[-1] + 1

    r_data = np.concatenate([np.ones((num_bins, 1)), np.eye(num_bins)], axis=1)

    # type level
    # This t_data is just encoding each with its own dummy (with bias terms)
    t_data = np.concatenate([np.ones((4, 1)), np.eye(4)], axis=1)

    # post level
    p_stories = np.empty((len(items),))
    p_subreddits = np.digitize(counts, subreddit_bins) - 1
    p_types = np.empty((len(items),))
    y = np.empty((len(items),))

    num_p_indep = 7
    p_data = np.zeros((len(items), num_p_indep))

    # set bias
    p_data[:, 0] = 1

    for i, n in enumerate(items):

        # post-level
        isNews = "isFakeStory" in n["r"]["reviewRating"]
        news_id = n["p"]["id"]
        # subscribers = max(n["p"]["subscribers"], 0)

        # story-level
        story_id = n["r"]["uid"]

        claim_title = n["r"]["claimReviewed"]
        p_countries = n["r"]["countries"]
        review_author = n["r"]["reviewAuthor"]["authorURL"]

        p_stories[i] = sid_indices[story_id]

        for c in p_countries:
            s_data[sid_indices[story_id], 1 + country_indices[c]] = 1

        author_indep_index = 1 + len(countries) + author_indices[review_author]
        s_data[sid_indices[story_id], author_indep_index] = 1

        story_claim_titles[sid_indices[story_id]] = claim_title

        num_cmts, cmts = comments[news_id]
        c_body_lens = []
        c_ups = []
        c_downs = []
        unique_authors = set()
        for c in cmts:
            # skip if comment not created in first _ mins
            if c["delta_seconds"] > minutes * 60:
                continue

            c_minutes = c["delta_seconds"] / 60
            if c_minutes <= minutes:  # cmts in first _ mins
                p_data[i, 6] += 1

            c_body_lens.append(c["body_len"])
            c_ups.append(c["ups"])
            c_downs.append(c["downs"])

            if c["author"]:
                unique_authors.add(c["author"])

        if isNews:
            p_types[i] = 1 if n["r"]["reviewRating"]["isFakeStory"] else 0
        else:
            p_types[i] = 3 if n["r"]["reviewRating"]["isFakeClaim"] else 2

        y[i] = num_cmts
        p_data[i, 1] = np.mean(c_body_lens) if c_body_lens else 0.0
        p_data[i, 2] = np.std(c_body_lens) if c_body_lens else 0.0
        p_data[i, 3] = np.mean(c_ups) if c_ups else 0.0
        p_data[i, 4] = np.std(c_ups) if c_ups else 0.0
        p_data[i, 5] = len(unique_authors) if unique_authors else 0.0

    # Adjust p_data

    # normalize avg net upvotes by std net upvotes:
    p_data[:, 3] = p_data[:, 3] / (p_data[:, 4] + 1)  # add 1 to avoid div by 0

    # select relevant indep vars
    # bias, num initial comments
    p_data = p_data[:, (0, 6)]

    # if comments only, prune further:
    if comments_only:
        p_data = p_data[:, (0, 1)]

    data_tuple = (p_data, t_data, s_data, r_data, y)
    lookup_tuple = (p_types, p_stories, p_subreddits)
    label_tuple = (countries, authors, story_claim_titles, subreddit_bins)

    return data_tuple, lookup_tuple, label_tuple


def transform_data(original_p_data, comments_only=False):
    p_data = original_p_data.copy()
    n_indeps = original_p_data.shape[1]

    # num initial comments
    p_data[:, 1] = np.log(p_data[:, 1] + 1)

    # num authors
    # p_data[:, 3] = np.log(p_data[:, 3] + 1)

    # avg comment len
    # p_data[:, 1] = np.log(p_data[:, 1] + 1)

    if n_indeps < 2:
        # num subscribers
        p_data[:, 2] = np.log(p_data[:, 2] + 1)

    return p_data


def split_and_prep_data(
    original_p_data,
    p_data,
    t_data,
    s_data,
    r_data,
    y,
    p_types,
    p_stories,
    p_subreddits,
    train_frac=0.7,
):
    num_obs = p_data.shape[0]
    idx = np.random.binomial(1, train_frac, size=num_obs).astype(bool)

    original_p_data_train = original_p_data[idx, :]
    original_p_data_test = original_p_data[~idx, :]

    p_data_train = p_data[idx, :]
    p_data_test = p_data[~idx, :]

    y_train = y[idx]
    y_test = y[~idx]

    p_types_train = p_types[idx]
    p_types_test = p_types[~idx]

    p_stories_train = p_stories[idx]
    p_stories_test = p_stories[~idx]

    p_subreddits_train = p_subreddits[idx]
    p_subreddits_test = p_subreddits[~idx]

    # convert everything to tensors
    # data
    original_p_data_train = torch.Tensor(original_p_data_train).double()
    original_p_data_test = torch.Tensor(original_p_data_test).double()
    p_data_train = torch.Tensor(p_data_train).double()
    p_data_test = torch.Tensor(p_data_test).double()
    t_data = torch.Tensor(t_data).double()
    s_data = torch.Tensor(s_data).double()
    r_data = torch.Tensor(r_data).double()

    y_train = torch.Tensor(y_train).double()
    y_test = torch.Tensor(y_test).double()

    # types
    p_types_train = torch.Tensor(p_types_train).long()
    p_types_test = torch.Tensor(p_types_test).long()
    p_stories_train = torch.Tensor(p_stories_train).long()
    p_stories_test = torch.Tensor(p_stories_test).long()
    p_subreddits_train = torch.Tensor(p_subreddits_train).long()
    p_subreddits_test = torch.Tensor(p_subreddits_test).long()

    train_data = (
        original_p_data_train,
        p_data_train,
        y_train,
        p_types_train,
        p_stories_train,
        p_subreddits_train,
    )
    test_data = (
        original_p_data_test,
        p_data_test,
        y_test,
        p_types_test,
        p_stories_test,
        p_subreddits_test,
    )

    unsplit_data = (t_data, s_data, r_data)

    return train_data, test_data, unsplit_data
