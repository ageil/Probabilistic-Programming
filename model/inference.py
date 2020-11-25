from time import time

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.optim as optim
from pyro.infer import SVI, JitTrace_ELBO


def run_svi(
    model,
    guide,
    train_data,
    unsplit_data,
    subsample=False,
    num_iters=10000,
    lr=1e-2,
    zero_inflated=False,
):
    p_data, y, p_types, p_stories, p_subreddits = train_data

    t_data, s_data, r_data = unsplit_data

    if subsample:
        p_data = p_data[:250]
        y = y[:250]
        p_types = p_types[:250]
        p_stories = p_stories[:250]
        p_subreddits = p_subreddits[:250]

    svi = SVI(
        model, guide, optim.ClippedAdam({"lr": lr}), loss=JitTrace_ELBO()
    )

    pyro.clear_param_store()
    losses = np.zeros((num_iters,))

    start_time = time()

    for i in range(num_iters):
        elbo = svi.step(
            p_data,
            t_data,
            s_data,
            r_data,
            y,
            p_types,
            p_stories,
            p_subreddits,
            zero_inflated,
        )
        losses[i] = elbo
        if i % 100 == 99:
            elapsed = time() - start_time
            remaining = (elapsed / (i + 1)) * (num_iters - i)
            print(
                f"Iter {i+1}/{num_iters}"
                "\t||\t"
                "Elbo loss:"
                f"{elbo:.2e}"
                "\t||\t"
                "Time Elapsed:"
                f"{int(elapsed) // 60:02}:{int(elapsed) % 60:02}"
                "\t||\t"
                f"Est Remaining:"
                f"{int(remaining) // 60:02}:{int(remaining) % 60:02}",
                end="\r",
                flush=True,
            )
    return svi, losses


def plot_losses(losses, log_scale=True, skip_first=0):
    plt.plot(range(len(losses[skip_first:])), np.array(losses[skip_first:]))
    plt.xlabel("Iteration")
    plt.ylabel("ELBO Loss")
    if log_scale:
        plt.yscale("log")
    plt.title("Learning Curve")
    plt.show()
