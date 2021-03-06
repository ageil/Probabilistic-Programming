import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints


def type_model(
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

    # type priors
    alpha_loc = torch.zeros((num_p_indeps, num_t_indeps), dtype=torch.float64)
    alpha_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, num_t_indeps), dtype=torch.float64
    )

    if zero_inflated:
        eta_gate_loc = torch.zeros(
            (num_p_indeps, num_t_indeps), dtype=torch.float64
        )
        eta_gate_scale = coef_scale_prior * torch.ones(
            (num_p_indeps, num_t_indeps), dtype=torch.float64
        )

    with pyro.plate("p_indep", num_p_indeps, dim=-2):

        # Type Level
        with pyro.plate("t_indep", num_t_indeps, dim=-1):
            eta = pyro.sample("eta", dist.Normal(alpha_loc, alpha_scale))
            if zero_inflated:
                eta_gate = pyro.sample(
                    "eta_gate", dist.Normal(eta_gate_loc, eta_gate_scale)
                )

        with pyro.plate("type", num_types, dim=-1) as t:
            phi_loc = torch.matmul(
                eta, t_data[t, :].T
            )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

            phi = pyro.sample("phi", dist.Normal(phi_loc, coef_scale_prior))

            if zero_inflated:
                phi_gate_loc = torch.matmul(
                    eta_gate, t_data[t, :].T
                )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

                phi_gate = pyro.sample(
                    "phi_gate", dist.Normal(phi_gate_loc, coef_scale_prior)
                )

    # for each post,
    # use the correct set of coefficients to run our post-level regression
    with pyro.plate("post", num_posts, dim=-1) as p:
        t = p_types[p]

        # indep vars for this post
        indeps = p_data[p, :]

        t_coefs = phi[:, t]  # (num_p_indeps,num_posts)

        type_level_products = torch.mul(
            t_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

        # calculate the mean: desired shape (num_posts, 1)
        mu = (type_level_products).sum(
            dim=0
        )  # (num_p_indeps, num_posts).sum(over indeps)

        # defining response dist
        if zero_inflated:
            t_coefs_gate = phi_gate[:, t]  # (num_p_indeps,num_posts)

            type_level_products_gate = torch.mul(
                t_coefs_gate, indeps.T
            )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

            # calculate the mean: desired shape (num_posts, 1)
            gate = torch.nn.Sigmoid()(
                (type_level_products_gate).sum(dim=0)
            )  # (num_p_indeps, num_posts).sum(over indeps)

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


def type_guide(
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

    # type level:
    eta_loc = pyro.param(
        "eta_loc",
        torch.zeros((num_p_indeps, num_t_indeps), dtype=torch.float64),
    )
    eta_scale = pyro.param(
        "eta_scale",
        coef_scale_prior
        * torch.ones((num_p_indeps, num_t_indeps), dtype=torch.float64),
        constraint=constraints.positive,
    )

    phi_scale = pyro.param(
        "phi_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all types.

    if zero_inflated:
        # type level:
        eta_gate_loc = pyro.param(
            "eta_gate_loc",
            torch.zeros((num_p_indeps, num_t_indeps), dtype=torch.float64),
        )
        eta_gate_scale = pyro.param(
            "eta_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, num_t_indeps), dtype=torch.float64),
            constraint=constraints.positive,
        )

        phi_gate_scale = pyro.param(
            "phi_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, 1), dtype=torch.float64),
            constraint=constraints.positive,
        )  # share among all types.

    with pyro.plate("p_indep", num_p_indeps, dim=-2):

        # type level
        with pyro.plate("t_indep", num_t_indeps, dim=-1):
            eta = pyro.sample("eta", dist.Normal(eta_loc, eta_scale))
            if zero_inflated:
                eta_gate = pyro.sample(
                    "eta_gate", dist.Normal(eta_gate_loc, eta_gate_scale)
                )

        with pyro.plate("type", num_types, dim=-1) as t:
            phi_loc = torch.matmul(
                eta, t_data[t, :].T
            )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

            pyro.sample("phi", dist.Normal(phi_loc, phi_scale))

            if zero_inflated:
                phi_gate_loc = torch.matmul(
                    eta_gate, t_data[t, :].T
                )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

                pyro.sample(
                    "phi_gate", dist.Normal(phi_gate_loc, phi_gate_scale)
                )
