from complete_model import complete_guide, complete_model
from data_proc import (
    load_raw_data,
    processData,
    split_and_prep_data,
    transform_data,
)
from evaluation import evaluate, get_samples
from inference import run_svi
from model_utils import get_y_pred
from post_model import post_guide, post_model
from story_model import story_guide, story_model
from subreddit_model import subreddit_guide, subreddit_model
from type_model import type_guide, type_model


class ModelManager:
    def __init__(self):
        self.model_dict = {
            "post": post_model,
            "type": type_model,
            "subreddit": subreddit_model,
            "story": story_model,
            "complete": complete_model,
        }
        self.guide_dict = {
            "post": post_guide,
            "type": type_guide,
            "subreddit": subreddit_guide,
            "story": story_guide,
            "complete": complete_guide,
        }
        self.results = None

    def load_data(self):
        comments, corrections, news = load_raw_data()
        data_tuple, lookup_tuple, label_tuple = processData(
            news + corrections, comments
        )

        original_p_data, t_data, s_data, r_data, y = data_tuple
        p_types, p_stories, p_subreddits = lookup_tuple
        countries, authors, story_claim_titles, subreddits = label_tuple

        self.original_p_data = original_p_data
        self.t_data = t_data
        self.s_data = s_data
        self.r_data = r_data
        self.y = y
        self.p_types = p_types
        self.p_stories = p_stories
        self.p_subreddits = p_subreddits
        self.countries = countries
        self.authors = authors
        self.story_claim_titles = story_claim_titles
        self.subreddits = subreddits

    def transform_data(self):
        self.p_data = transform_data(self.original_p_data)

    def split_data(self, train_frac=0.7):
        train_data, test_data, unsplit_data = split_and_prep_data(
            self.original_p_data,
            self.p_data,
            self.t_data,
            self.s_data,
            self.r_data,
            self.y,
            self.p_types,
            self.p_stories,
            self.p_subreddits,
            train_frac=train_frac,
        )

        self.train_data = train_data
        self.test_data = test_data
        self.unsplit_data = unsplit_data

        orig_p_data, p_data, y, p_types, p_stories, p_sub = self.train_data
        self.original_p_data_train = orig_p_data
        self.p_data_train = p_data
        self.y_train = y
        self.p_types_train = p_types
        self.p_stories_train = p_stories
        self.p_subreddits_train = p_sub

        orig_p_data, p_data, y, p_types, p_stories, p_sub = self.test_data
        self.original_p_data_test = orig_p_data
        self.p_data_test = p_data
        self.y_test = y
        self.p_types_test = p_types
        self.p_stories_test = p_stories
        self.p_subreddits_test = p_sub

        self.t_data, self.s_data, self.r_data = self.unsplit_data

    def run_svi(self, model_name, num_iters=1000, lr=1e-2, zero_inflated=True):
        _, type_losses = run_svi(
            self.model_dict[model_name],
            self.guide_dict[model_name],
            self.train_data,
            self.unsplit_data,
            num_iters=num_iters,
            lr=lr,
            zero_inflated=zero_inflated,
        )

        return type_losses

    def get_y_pred(self, use_train=True):
        if use_train:
            return get_y_pred(
                self.p_data_train,
                self.t_data,
                self.s_data,
                self.r_data,
                self.p_types_train,
                self.p_stories_train,
                self.p_subreddits_train,
            )
        else:
            return get_y_pred(
                self.p_data_test,
                self.t_data,
                self.s_data,
                self.r_data,
                self.p_types_test,
                self.p_stories_test,
                self.p_subreddits_test,
            )

    def get_samples(self, model_name, zero_inflated=True, num_samples=1000):
        return get_samples(
            self.model_dict[model_name],
            self.guide_dict[model_name],
            self.p_data_train,
            self.t_data,
            self.s_data,
            self.r_data,
            None,
            self.p_types_train,
            self.p_stories_train,
            self.p_subreddits_train,
            zero_inflated,
            num_samples=num_samples,
        )

    def evaluate(self, y_pred, partition="train", model="type"):
        y_data = self.y_train if partition == "train" else self.y_test
        self.results = evaluate(
            self.results, y_data, y_pred, partition=partition, model=model
        )
