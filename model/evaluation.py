import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro.infer import Predictive
from pyro.ops.stats import quantile
from collections import defaultdict

P_INDEP_DICT = {1: "Comments in First Hour", 2: "Subscribers"}

LABELS = [
    "Factual News",
    "Fake News",
    "Review of Factual News",
    "Review of Fake News",
]


# only_type should be one of 0, 1, 2, or 3, corresponding to the type of post.
def plot_predictions(
    original_p_data,
    y,
    p_types,
    p_data_pred,
    y_pred,
    p_types_pred,
    indep=1,
    only_type=None,
    log_scale=True,
    filename="predictions.png",
    **scatter_kwargs,
):

    plt.figure(figsize=(12, 9))

    types = np.unique(p_types)

    for i, t in enumerate(types):
        if only_type is None or t == only_type:
            x_pred_t = p_data_pred[p_types_pred == t, indep]
            y_pred_t = y_pred[p_types_pred == t]

            x_t = original_p_data[p_types == t, indep]
            y_t = y[p_types == t]

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
                **scatter_kwargs,
            )
            plt.plot(
                x_pred_t[sorted_indices_pred],
                y_pred_t[sorted_indices_pred],
                c=color,
                ls=line_style,
                label=f"Predicted {type_label}",
            )

    plt.xlabel(P_INDEP_DICT[indep])
    plt.ylabel("Total Comments")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.legend()
    plt.savefig(f"../output/{filename}")
    plt.show()


def plot_predictions_by_subreddit(
    original_p_data,
    y,
    p_subreddits,
    p_data_pred,
    y_pred,
    p_subreddits_pred,
    indep=1,
    log_scale=True,
    **scatter_kwargs,
):

    plt.figure(figsize=(12, 9))

    subreddits = np.unique(p_subreddits)

    rows = int(np.ceil(np.sqrt(len(subreddits))))
    cols = int(np.ceil(len(subreddits) / rows))

    x_min = 1e-1
    x_max = max(original_p_data[:, indep])
    y_min = 1e-1
    y_max = max(y)

    for i, r in enumerate(subreddits):
        x_pred_r = p_data_pred[p_subreddits_pred == r, indep]
        y_pred_r = y_pred[p_subreddits_pred == r]

        x_r = original_p_data[p_subreddits == r, indep]
        y_r = y[p_subreddits == r]

        sorted_indices_pred = np.argsort(x_pred_r)

        subreddit_label = f"{i+1}th 1/{len(subreddits)}-ile"

        plt.subplot(rows, cols, i + 1)
        plt.scatter(
            x_r,
            y_r,
            s=12,
            label=f"Observed {subreddit_label}",
            **scatter_kwargs,
        )
        plt.plot(
            x_pred_r[sorted_indices_pred],
            y_pred_r[sorted_indices_pred],
            label=f"Predicted {subreddit_label}",
        )

        plt.xlabel(P_INDEP_DICT[indep])
        plt.ylabel("Total Comments")

        if log_scale:
            plt.yscale("log")
            plt.xscale("log")
        # plt.legend()

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(subreddit_label)
    plt.suptitle("Predictions by Subreddit")
    plt.tight_layout()


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


def plot_ppc_grid(samples, y):
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
        "Predicted zeros (%)",
        "Predicted max",
        "Predicted variance",
        "Predicted non-zero mean",
    ]
    labels = ["fraction zeros", "max", "variance", "non-zero mean"]

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
    plt.legend(bbox_to_anchor=(1.05, 1.05), loc='lower left', borderaxespad=0.)
#    plt.legend(loc='lower center', bbox_to_anchor = (0,-0.1,1,1))
    plt.tight_layout()


# TODO plot by type.
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


def plot_pp_hdi(samples, p_data, y, hdi_prob=0.99):
    x_data = p_data[:, 1].detach().numpy()
    y_data = samples["obs"][0, :, :]
    sorted_indices = np.argsort(x_data)
    x_sorted = x_data[sorted_indices]
    y_sorted = y_data[:, sorted_indices]

    az.plot_hdi(
        x_sorted,
        y_sorted,
        color="k",
        plot_kwargs={"ls": "--"},
        hdi_prob=hdi_prob,
        fill_kwargs={"label": f"{100*hdi_prob}% HDI"},
    )
    plt.yscale("log")
    plt.plot(x_sorted, np.mean(y_sorted, axis=0), "C6", label="Mean Pred")
    plt.scatter(p_data[:, 1], y, s=12, alpha=0.1, label="Observed")
    plt.title("HDI Posterior Predictive Plot")
    plt.xlabel("log num comments 1st hour")
    plt.ylabel("num total comments")
    plt.legend()
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
        ["Real News", "Fake News"],
        ["Correction on Real", "Correction on Fake"],
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
    plt.xlabel("Log Total Comments (Engagement)")
    plt.tight_layout()


def MAE(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.mean(np.abs(y - y_hat))


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


def evaluate(results, y, y_pred, partition='train', model='post'):
    if results is None:
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    results[partition][model]['R^2'] = R2(y, y_pred)
    results[partition][model]['R^2_log'] = R2(np.log(y+1), np.log(y_pred+1))
    results[partition][model]['MAE'] = MAE(y, y_pred)
    results[partition][model]['MAE_log'] = MAE(np.log(y+1), np.log(y_pred+1))
    return results
