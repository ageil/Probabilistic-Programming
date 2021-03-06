from collections import defaultdict

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro.infer import Predictive
from pyro.ops.stats import quantile
from tabulate import tabulate

P_INDEP_DICT = {1: "Comments in First Hour", 2: "Subscribers"}

LABELS = [
    "Factual News",
    "Fake News",
    "Review of Factual News",
    "Review of Fake News",
]


# only_type should be one of 0, 1, 2, or 3, corresponding to the type of post.
def plot_predictions(
    y,
    original_p_data,
    p_types,
    y_pred,
    p_data_pred=None,
    p_types_pred=None,
    indep=1,
    only_type=None,
    log_scale=True,
    filename="predictions.png",
    alpha=0.2,
    **scatter_kwargs,
):
    if p_data_pred is None:
        p_data_pred = original_p_data
    if p_types_pred is None:
        p_types_pred = p_types

    plt.figure(figsize=(12, 9))

    types = np.unique(p_types)

    for i, t in enumerate(types):
        if only_type is None or t == only_type:
            x_pred_t = p_data_pred[p_types_pred == t, indep] + 1
            y_pred_t = y_pred[p_types_pred == t] + 1

            x_t = original_p_data[p_types == t, indep] + 1
            y_t = y[p_types == t] + 1

            sorted_indices_pred = np.argsort(x_pred_t)

            type_label = LABELS[int(t)]

            color = "tab:green" if i % 2 == 0 else "tab:red"
            line_style = "-" if i < 2 else "dotted"
            scatter_style = "o" if i < 2 else "x"
            face_colors = "none" if i < 2 else color

            plt.scatter(
                x_t,
                y_t,
                s=16,
                edgecolors=color,
                facecolors=face_colors,
                marker=scatter_style,
                label=f"Observed {type_label}",
                alpha=alpha,
                **scatter_kwargs,
            )
            plt.plot(
                x_pred_t[sorted_indices_pred],
                y_pred_t[sorted_indices_pred],
                c=color,
                ls=line_style,
                label=f"Predicted {type_label}",
            )

    plt.xlabel(P_INDEP_DICT[indep] + " (+1)")
    plt.ylabel("Future Comments (+1)")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.ylim(1 - 1e-3, torch.max(y).detach().numpy() * 2)
    plt.legend()
    plt.savefig(f"../output/{filename}")
    plt.show()


def plot_predictions_by_subreddit(
    y,
    original_p_data,
    p_subreddits,
    y_pred,
    p_data_pred=None,
    p_subreddits_pred=None,
    indep=1,
    log_scale=True,
    filename="predictions.png",
    alpha=0.2,
    **scatter_kwargs,
):
    if p_data_pred is None:
        p_data_pred = original_p_data
    if p_subreddits_pred is None:
        p_subreddits_pred = p_subreddits

    plt.figure(figsize=(12, 9))

    subreddits = np.unique(p_subreddits)

    rows = int(np.ceil(np.sqrt(len(subreddits))))
    cols = int(np.ceil(len(subreddits) / rows))

    x_min = 1
    x_max = max(original_p_data[:, indep])
    y_min = 1
    y_max = max(y)

    for i, r in enumerate(subreddits):
        x_pred_r = p_data_pred[p_subreddits_pred == r, indep] + 1
        y_pred_r = y_pred[p_subreddits_pred == r] + 1

        x_r = original_p_data[p_subreddits == r, indep] + 1
        y_r = y[p_subreddits == r] + 1

        sorted_indices_pred = np.argsort(x_pred_r)

        subreddit_label = f"{i+1}th 1/{len(subreddits)}-ile"

        plt.subplot(rows, cols, i + 1)
        plt.scatter(
            x_r,
            y_r,
            s=12,
            label=f"Observed {subreddit_label}",
            alpha=alpha,
            **scatter_kwargs,
        )
        plt.plot(
            x_pred_r[sorted_indices_pred],
            y_pred_r[sorted_indices_pred],
            label=f"Predicted {subreddit_label}",
        )

        plt.xlabel(P_INDEP_DICT[indep] + " (+1)")
        plt.ylabel("Future Comments (+1)")

        if log_scale:
            plt.yscale("log")
            plt.xscale("log")
        # plt.legend()

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, 2 * y_max)
        plt.title(subreddit_label)
    plt.suptitle("Predictions by Subreddit")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"../output/{filename}", bbox_inches="tight")


def get_samples(model, guide, *args, num_samples=1000, detach=True):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    if detach:
        svi_samples = {
            k: v.reshape((1, num_samples, -1)).detach().cpu().numpy()
            for k, v in predictive(*args).items()
        }
    else:
        svi_samples = {
            k: v.reshape((1, num_samples, -1))
            for k, v in predictive(*args).items()
        }
    return svi_samples


def get_quantiles(samples, param, quantiles=(0.1, 0.5, 0.9)):
    qs = quantile(torch.squeeze(samples[param], dim=0), quantiles, dim=0)
    return qs


def gather_az_inference_data(svi_samples, y):
    inf_data = az.convert_to_inference_data(
        {"obs": svi_samples["obs"]}, group="posterior_predictive"
    )
    inf_data.add_groups(
        {
            "posterior": {
                site: samples
                for site, samples in svi_samples.items()
                if site != "obs"
            }
        }
    )
    inf_data.add_groups({"observed_data": {"obs": y.reshape(1, 1, -1)}})
    return inf_data


def plot_pp_cdf(inf_data, y):
    # calculate
    y = np.array(y)
    uniq = np.unique(y).astype(int)
    cum_density = []

    for i in uniq:
        density = (y <= i).mean()
        cum_density.append((i, density))

    cum_density = np.array(cum_density)

    # plot cdf
    plt.figure(figsize=(9, 12))
    az.plot_ppc(inf_data, kind="cumulative")
    plt.xscale("log")
    plt.plot(
        cum_density[:, 0],
        cum_density[:, 1],
        c="tab:red",
        label="Observed obs (manual)",
    )
    plt.set_cmap("jet")
    plt.legend(loc="lower right")
    plt.show()


def plot_pp_pdf(inf_data, y):
    # calculate
    y = np.array(y)
    uniq = np.unique(y).astype(int)
    densities = []

    for i in uniq:
        densities.append((i, (y == i).mean()))

    densities = np.array(densities)

    # plot pdf
    plt.figure(figsize=(9, 12))
    az.plot_ppc(inf_data, kind="kde")
    plt.xscale("log")
    plt.plot(
        densities[:, 0],
        densities[:, 1],
        c="tab:red",
        label="Observed obs (manual)",
    )
    plt.set_cmap("jet")
    plt.legend(loc="lower right")
    plt.show()


# func should calculate the ppc along axis 0.
def plot_ppc(
    svi_samples,
    y,
    func,
    label,
    title=None,
    log_stats=True,
    log_freqs=False,
    legend=True,
    show=True,
):
    y = np.array(y)
    obs_per_draw = len(y)
    stats = func(svi_samples["obs"].reshape(-1, obs_per_draw).T)
    obs_stat = func(y)
    mean_stat = np.mean(stats)

    if log_stats:
        stats = np.log(stats)
        obs_stat = np.log(obs_stat)
        mean_stat = np.log(mean_stat)

    x_label = f"log({label})" if log_stats else label
    title = label if title is None else title

    plt.hist(stats, alpha=0.5, label="Hallucinated")
    plt.axvline(obs_stat, color="tab:red", label="Observed")
    plt.axvline(mean_stat, color="tab:green", label="Mean Hallucinated")
    plt.title(f"{title}")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    if log_freqs:
        plt.yscale("log")
    if legend:
        plt.legend()
    if show:
        plt.show()


def plot_ppc_grid(samples, y, filename="ppc.png"):
    def zero_func(x):
        return (x == 0).mean(axis=0)

    def max_func(x):
        return np.max(x, axis=0)

    def var_func(x):
        return np.var(x, axis=0)

    def mean_func(x):
        return np.mean(np.log(x + 1), axis=0)

    funcs = [zero_func, max_func, mean_func, var_func]
    titles = [
        "Predicted Zeros (%)",
        "Predicted Max",
        "Predicted Variance",
        "Predicted Non-Zero Mean",
    ]
    labels = ["Fraction Zeros", "Max", "Variance", "Non-Zero Mean"]

    plt.figure(figsize=(12, 8))
    for i, (func, title, label) in enumerate(zip(funcs, titles, labels)):
        plt.subplot(2, 2, i + 1)
        log_stats = (i == 1) or (i == 3)
        plot_ppc(
            samples,
            y,
            func,
            label=label,
            title=title,
            legend=False,
            show=False,
            log_stats=log_stats,
        )
    plt.legend(
        bbox_to_anchor=(1.05, 1.05), loc="lower left", borderaxespad=0.0
    )
    plt.tight_layout()
    plt.savefig(f"../output/{filename}", bbox_inches="tight")


def plot_residuals(y, y_pred, title="Residuals (Obs - Pred)"):
    residuals = y - y_pred

    plt.hist(residuals)
    plt.yscale("log")
    plt.title(title)
    plt.show()


def plot_residuals_by_type(y, y_pred, p_types):
    y = np.array(y)
    y_pred = np.array(y_pred)
    for t in np.unique(p_types):
        t = int(t)
        y_t = y[p_types == t]
        y_pred_t = y_pred[p_types == t]
        plt.title()
        plot_residuals(
            y_t,
            y_pred_t,
            title=f"Residuals (Obs - Pred): {np.array(LABELS)[t]}",
        )


def plot_pp_hdi(
    samples,
    original_p_data,
    y,
    hdi_prob=0.99,
    log_scale=True,
    limit=False,
    filename="hdi.png",
):
    original_x_data = original_p_data[:, 1].detach().numpy()
    y_data = samples["obs"]
    sorted_indices = np.argsort(original_x_data)
    original_x_sorted = original_x_data[sorted_indices]
    y_sorted = y_data[:, :, sorted_indices]

    hdi_data = az.hdi(y_sorted, hdi_prob=hdi_prob)

    x_unique = np.unique(original_x_sorted)

    # take the largest bounds across all observations for this x.
    y_bounds = np.empty((x_unique.shape[0], 2))

    # mean across all observations for corresponding x.
    y_means = np.empty((x_unique.shape[0]))

    for i, x in enumerate(x_unique):
        y_bounds[i, 1] = np.max(hdi_data[original_x_sorted == x, 1])
        y_bounds[i, 0] = np.min(hdi_data[original_x_sorted == x, 0])
        y_means[i] = np.mean(
            y_sorted[0, :, original_x_sorted == x], axis=(0, 1)
        )

    plt.figure(figsize=(12, 8))

    plt.fill_between(
        x_unique + 1,
        y_bounds[:, 0] + 1,
        y_bounds[:, 1] + 1,
        color="tab:gray",
        label=f"{100*hdi_prob}% HDI",
    )

    if log_scale:
        plt.xscale("log")
        plt.yscale("log")
    plt.plot(x_unique + 1, y_means + 1, "C6", label="Mean Pred")
    if limit:
        plt.xlim(-1, 10)
        plt.ylim(-1, 50)
    plt.scatter(original_x_data + 1, y + 1, s=12, alpha=0.1, label="Observed")
    plt.title("HDI Posterior Predictive Plot")
    plt.xlabel("Comments in First Hour (+1)")
    plt.ylabel("Future Comments (+1)")
    plt.legend()
    plt.savefig(f"../output/{filename}")
    plt.show()


def plot_expectations(y, p_types):
    real_news_posts = y[p_types == 0]
    fake_news_posts = y[p_types == 1]
    real_news_correction = y[p_types == 2]
    fake_news_correction = y[p_types == 3]
    data = [
        [real_news_posts, fake_news_posts],
        [real_news_correction, fake_news_correction],
    ]
    labels = [
        ["Factual news", "Fake news"],
        ["Correction on factual", "Correction on fake"],
    ]
    colors = [["tab:green", "tab:red"], ["tab:green", "tab:red"]]
    plt.figure(figsize=(15, 6))
    for i, (data_list, label_list, color_list) in enumerate(
        zip(data, labels, colors)
    ):

        plt.subplot(2, 1, i + 1)

        for d, l, c in zip(data_list, label_list, color_list):
            hist, bins = np.histogram(d + 1, bins=20)
            logbins = np.logspace(
                np.log10(bins[0]), np.log10(bins[-1]), len(bins)
            )
            plt.hist(
                d + 1, alpha=0.25, label=l, bins=logbins, density=True, color=c
            )
            plt.xscale("log")
            #         plt.xlim(0, 6000)
            plt.ylim(1e-7, 1)
            plt.xlim(1, 1e4)
            plt.axvline(d.mean(), label=f"{l} (Expected)", c=c)
            plt.text(
                np.exp(np.log(d.mean()) + 0.06),
                0.3,
                f"{np.round(d.mean(), 2)}",
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("Probability")
        plt.legend()
    plt.suptitle("Type Distributions and Expected Values")
    plt.xlabel("Log Future Comments (Engagement)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("../output/expectations.png", bbox_inches="tight")


def MAE(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.mean(np.abs(y - y_hat))


def MAElog(y, y_hat):
    return MAE(np.log(y[y > 0] + 1), np.log(y_hat[y > 0] + 1))


def MSE(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.mean((y - y_hat) ** 2)


def R2(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    SSE = np.sum((y - y_hat) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SSE / TSS
    return R2


def R2log(y, y_hat):
    return R2(np.log(y[y > 0] + 1), np.log(y_hat[y > 0] + 1))


def evaluate(results, y, y_pred, partition="train", model="post"):
    if results is None:
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    results[partition][model]["R^2"] = np.round(R2(y, y_pred), 2)
    results[partition][model]["R^2 log non-zero"] = np.round(
        R2log(y, y_pred), 2
    )
    return results


def print_results(results, partition="train"):
    model_results = results[partition]
    models = list(model_results.keys())

    stats = list(model_results["type"].keys())
    vals = [stats] + [list(model_results[model].values()) for model in models]
    vals = np.asarray(vals).T

    headers = [""] + models
    print(tabulate(vals, headers=headers))
