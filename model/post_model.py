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
    num_types, num_t_indeps = t_data.shape

    # shared prior
    gamma_loc = torch.zeros((num_p_indeps, 1), dtype=torch.float64)

    with pyro.plate("p_indep", num_p_indeps, dim=-2):
        gamma = pyro.sample("gamma", dist.Normal(gamma_loc, coef_scale_prior))

    # Gate
    if zero_inflated[0]:
        gate = pyro.sample(
            "gate",
            dist.Beta(
                torch.ones((1,), dtype=torch.float64),
                torch.ones((1,), dtype=torch.float64),
            ),
        )

    # for each post,
    # use the correct set of coefficients to run our post-level regression
    with pyro.plate("post", num_posts, dim=-1) as p:

        # indep vars for this post
        indeps = p_data[p, :]

        coefs = gamma.repeat((1, num_posts))  # (num_p_indeps,num_posts)

        products = torch.mul(
            coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

        # calculate the mean: desired shape (num_posts, 1)
        mu = (products).sum(
            dim=0
        )  # (num_p_indeps, num_posts).sum(over indeps)

        # defining response dist
        if zero_inflated[0]:
            response_dist = dist.ZeroInflatedPoisson(
                rate=torch.exp(mu), gate=gate.flatten()
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
    num_types, num_t_indeps = t_data.shape

    # define a prior for our regression variables
    gamma_loc = pyro.param(
        "gamma_loc", torch.zeros((num_p_indeps, 1), dtype=torch.float64)
    )

    gamma_scale = pyro.param(
        "gamma_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all types.

    with pyro.plate("p_indep", num_p_indeps, dim=-2):
        pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale))

    # Gate, shared across all posts
    if zero_inflated[0]:
        gate_alpha = pyro.param(
            "gate_alpha",
            2.0 * torch.ones((1,), dtype=torch.float64),
            constraint=constraints.positive,
        )
        gate_beta = pyro.param(
            "gate_beta",
            2.0 * torch.ones((1,), dtype=torch.float64),
            constraint=constraints.positive,
        )
        pyro.sample("gate", dist.Beta(gate_alpha, gate_beta))
