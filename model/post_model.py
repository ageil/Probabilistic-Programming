import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints


def post_model(
    p_data,
    t_data,
    s_data,
    r_data,
    y,
    p_types,
    p_stories,
    p_subreddits,
    zero_inflated,
):
    coef_scale_prior = 0.1

    num_posts, num_p_indeps = p_data.shape

    # shared prior
    gamma_loc = torch.zeros((num_p_indeps, 1), dtype=torch.float64)

    if zero_inflated:
        gamma_gate_loc = torch.zeros((num_p_indeps, 1), dtype=torch.float64)

    with pyro.plate("p_indep", num_p_indeps, dim=-2):
        gamma = pyro.sample("gamma", dist.Normal(gamma_loc, coef_scale_prior))

        if zero_inflated:
            gamma_gate = pyro.sample(
                "gamma_gate", dist.Normal(gamma_gate_loc, coef_scale_prior)
            )

    # for each post,
    # use the correct set of coefficients to run our post-level regression
    with pyro.plate("post", num_posts, dim=-1) as p:

        # indep vars for this post
        indeps = p_data[p, :]

        mu = torch.matmul(
            indeps, gamma
        ).flatten()  # ( num_posts, num_p_indeps) x (num_p_indeps, 1)

        # defining response dist
        if zero_inflated:
            gate = torch.nn.Sigmoid()(
                torch.matmul(indeps, gamma_gate).flatten()
            )  # ( num_posts, num_p_indeps) x (num_p_indeps, 1)

            response_dist = dist.ZeroInflatedPoisson(
                rate=torch.exp(mu), gate=gate
            )
        else:
            response_dist = dist.Poisson(rate=torch.exp(mu))

        # sample
        if y is None:
            pyro.sample("obs", response_dist, obs=y)
        else:
            pyro.sample("obs", response_dist, obs=y[p])


def post_guide(
    p_data,
    t_data,
    s_data,
    r_data,
    y,
    p_types,
    p_stories,
    p_subreddits,
    zero_inflated,
):
    coef_scale_prior = 0.1

    num_posts, num_p_indeps = p_data.shape

    # define a prior for our regression variables
    gamma_loc = pyro.param(
        "gamma_loc", torch.zeros((num_p_indeps, 1), dtype=torch.float64)
    )

    gamma_scale = pyro.param(
        "gamma_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all types.

    if zero_inflated:
        gamma_gate_loc = pyro.param(
            "gamma_gate_loc",
            torch.zeros((num_p_indeps, 1), dtype=torch.float64),
        )

        gamma_gate_scale = pyro.param(
            "gamma_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, 1), dtype=torch.float64),
            constraint=constraints.positive,
        )  # share among all types.

    with pyro.plate("p_indep", num_p_indeps, dim=-2):
        pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale))
        if zero_inflated:
            pyro.sample(
                "gamma_gate", dist.Normal(gamma_gate_loc, gamma_gate_scale)
            )
