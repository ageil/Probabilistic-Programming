import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pyro.infer import Predictive

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
    indep=4,
    only_type=None,
    log_scale=True,
    **scatter_kwargs,
):

    plt.figure(figsize=(12, 9))

    types = np.unique(p_types)

    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    color_choices = np.random.choice(colors, size=len(types), replace=False)

    for i, t in enumerate(types):
        if only_type is None or t == only_type:
            x_pred_t = p_data_pred[p_types_pred == t, indep]
            y_pred_t = y_pred[p_types_pred == t]

            x_t = original_p_data[p_types == t, indep]
            y_t = y[p_types == t]

            sorted_indices_pred = np.argsort(x_pred_t)

            type_label = LABELS[int(t)]

            color = color_choices[i]
            plt.scatter(
                x_t,
                y_t,
                s=12,
                c=color,
                label=f"Actual {type_label}",
                **scatter_kwargs,
            )
            plt.plot(
                x_pred_t[sorted_indices_pred],
                y_pred_t[sorted_indices_pred],
                c=color,
                label=f"Predicted {type_label}",
            )
    
    plt.xlabel(P_INDEP_DICT[indep])
    plt.ylabel("Total Comments")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.legend()
    plt.show()

def get_samples(model, guide, *args, num_samples=1000):
    predictive = Predictive(model, guide=guide, num_samples=num_samples)
    svi_samples = {
        k: v.reshape((1, num_samples, -1)).detach().cpu().numpy()
        for k, v in predictive(*args).items()
    }
    return svi_samples


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
def plot_ppc(svi_samples, y, func, label, log_stats=True):
    y = np.array(y)
    obs_per_draw = len(y)
    stats = func(svi_samples["obs"].reshape(obs_per_draw, -1))
    obs_stat = func(y)
    mean_stat = np.mean(stats)

    if log_stats:
        stats = np.log(stats)
        obs_stat = np.log(obs_stat)
        mean_stat = np.log(mean_stat)

    x_label = f"log({label})" if log_stats else label

    plt.hist(stats, alpha=0.5, label="Hallucinated")
    plt.axvline(obs_stat, color="tab:red", label="Observed")
    plt.axvline(mean_stat, color="tab:green", label="Mean Hallucinated")
    plt.title(f"PPC of {label}")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# TODO plot by type.
def plot_residuals(y, y_pred):
    residuals = y - y_pred

    plt.hist(residuals)
    plt.yscale("log")
    plt.title("Residuals (Obs - Pred)")
    plt.show()
