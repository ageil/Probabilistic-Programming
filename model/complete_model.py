import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints


def get_centered(loc, p_groups):

    num_posts = len(p_groups)
    num_groups = loc.shape[1]

    # This centers w.r.t. the weighting found in the data.
    unique, group_counts = torch.unique(p_groups, return_counts=True)
    all_group_counts = torch.zeros((num_groups,), dtype=torch.float64)
    all_group_counts[unique] = group_counts.double()

    # loc shape: (num_p_indeps, num_groups)
    # group counts shape: (num_groups)
    # for each p_indep, take weighted mean according to group_counts

    # matrix vector mult.
    loc_means = (
        torch.matmul(loc, all_group_counts).reshape((-1, 1)) / num_posts
    )

    # This centers uniformly
    # loc_means = loc.mean(dim=-1, keepdim=True)

    # enforce the mean across groups to be 0.
    loc_centered = loc - loc_means

    return loc_centered


def complete_model(
    p_data, t_data, s_data, r_data, y, p_types, p_stories, p_subreddits
):
    coef_scale_prior = 0.1

    num_posts, num_p_indeps = p_data.shape
    num_types, num_t_indeps = t_data.shape
    num_stories, num_s_indeps = s_data.shape
    num_subreddits, num_r_indeps = r_data.shape

    # type priors
    alpha_loc = torch.zeros((num_p_indeps, num_t_indeps), dtype=torch.float64)
    alpha_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, num_t_indeps), dtype=torch.float64
    )

    # story priors
    beta_loc = torch.zeros((num_p_indeps, num_s_indeps), dtype=torch.float64)
    beta_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, num_s_indeps), dtype=torch.float64
    )

    # type priors
    tau_loc = torch.zeros((num_p_indeps, num_r_indeps), dtype=torch.float64)
    tau_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, num_r_indeps), dtype=torch.float64
    )

    # shared priors
    gamma_loc = torch.zeros((num_p_indeps, 1), dtype=torch.float64)
    gamma_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, 1), dtype=torch.float64
    )

    with pyro.plate("p_indep", num_p_indeps, dim=-2):

        # Type Level
        with pyro.plate("t_indep", num_t_indeps, dim=-1):
            eta = pyro.sample("eta", dist.Normal(alpha_loc, alpha_scale))

        with pyro.plate("type", num_types, dim=-1) as t:
            phi_loc = torch.matmul(
                eta, t_data[t, :].T
            )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

            phi_loc_centered = get_centered(phi_loc, p_types)

            phi = pyro.sample(
                "phi", dist.Normal(phi_loc_centered, coef_scale_prior)
            )

        # Story Level

        with pyro.plate("s_indep", num_s_indeps, dim=-1):
            beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale))

        with pyro.plate("story", num_stories, dim=-1) as s:
            theta_loc = torch.matmul(
                beta, s_data[s, :].T
            )  # (num_p_indeps, num_s_indeps) x (num_s_indeps, num_stories)

            theta_loc_centered = get_centered(theta_loc, p_stories)

            theta = pyro.sample(
                "theta", dist.Normal(theta_loc_centered, coef_scale_prior)
            )

        # Subreddit Level

        with pyro.plate("r_indep", num_r_indeps, dim=-1):
            tau = pyro.sample("tau", dist.Normal(tau_loc, tau_scale))

        with pyro.plate("subreddit", num_subreddits, dim=-1) as r:
            rho_loc = torch.matmul(
                tau, r_data[r, :].T
            )  # (num_p_indeps, num_r_indeps) x (num_r_indeps, num_subreddits)

            rho_loc_centered = get_centered(rho_loc, p_subreddits)

            rho = pyro.sample(
                "rho", dist.Normal(rho_loc_centered, coef_scale_prior)
            )

        # Shared
        gamma_dist = dist.Normal(gamma_loc, gamma_scale)
        gamma = pyro.sample("gamma", gamma_dist)

    # Gate

    with pyro.plate("type2", num_types, dim=-1):
        gate = pyro.sample(
            "gate",
            dist.Beta(
                torch.ones((num_types,), dtype=torch.float64),
                torch.ones((num_types,), dtype=torch.float64),
            ),
        )

    # for each post,
    # use the correct set of coefficients to run our post-level regression
    with pyro.plate("post", num_posts, dim=-1) as p:
        t = p_types[p]
        s = p_stories[p]
        r = p_subreddits[p]

        # indep vars for this post
        indeps = p_data[p, :]

        t_coefs = phi[:, t]  # (num_p_indeps,num_posts)
        s_coefs = theta[:, s]  # (num_p_indeps,num_posts)
        r_coefs = rho[:, r]  # (num_p_indeps,num_posts)
        shared_coefs = gamma.repeat((1, num_posts))  # (num_p_indeps,num_posts)

        type_level_products = torch.mul(
            t_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
        story_level_products = torch.mul(
            s_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
        subreddit_level_products = torch.mul(
            r_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
        shared_products = torch.mul(
            shared_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

        # calculate the mean: desired shape (num_posts, 1)
        mu = (
            subreddit_level_products
            + type_level_products
            + story_level_products
            + shared_products
        ).sum(
            dim=0
        )  # (num_p_indeps, num_posts).sum(over indeps)

        # sample
        if y is None:
            pyro.sample(
                "obs",
                dist.ZeroInflatedPoisson(
                    rate=torch.exp(mu), gate=gate.flatten()[t]
                ),
                obs=y,
            )
        else:
            pyro.sample(
                "obs",
                dist.ZeroInflatedPoisson(
                    rate=torch.exp(mu), gate=gate.flatten()[t]
                ),
                obs=y[p],
            )


def complete_guide(
    p_data, t_data, s_data, r_data, y, p_types, p_stories, p_subreddits
):
    coef_scale_prior = 0.1

    num_posts, num_p_indeps = p_data.shape
    num_types, num_t_indeps = t_data.shape
    num_stories, num_s_indeps = s_data.shape
    num_subreddits, num_r_indeps = r_data.shape

    # define a prior for our regression variables

    # type level:
    # The zeros and ones are the "alpha"
    # in the graphical model from the proposal
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
    # story level
    beta_loc = pyro.param(
        "beta_loc",
        torch.zeros((num_p_indeps, num_s_indeps), dtype=torch.float64),
    )
    beta_scale = pyro.param(
        "beta_scale",
        coef_scale_prior
        * torch.ones((num_p_indeps, num_s_indeps), dtype=torch.float64),
        constraint=constraints.positive,
    )

    theta_scale = pyro.param(
        "theta_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all stories.

    # subreddit level
    tau_loc = pyro.param(
        "tau_loc",
        torch.zeros((num_p_indeps, num_r_indeps), dtype=torch.float64),
    )
    tau_scale = pyro.param(
        "tau_scale",
        coef_scale_prior
        * torch.ones((num_p_indeps, num_r_indeps), dtype=torch.float64),
        constraint=constraints.positive,
    )

    rho_scale = pyro.param(
        "rho_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all subreddits.

    gamma_loc = pyro.param(
        "gamma_loc", torch.zeros((num_p_indeps, 1), dtype=torch.float64)
    )  # share among all.

    gamma_scale = pyro.param(
        "gamma_scale",
        coef_scale_prior * torch.ones((num_p_indeps, 1), dtype=torch.float64),
        constraint=constraints.positive,
    )  # share among all.

    with pyro.plate("p_indep", num_p_indeps, dim=-2):

        # type level

        with pyro.plate("t_indep", num_t_indeps, dim=-1):
            eta = pyro.sample("eta", dist.Normal(eta_loc, eta_scale))

        with pyro.plate("type", num_types, dim=-1) as t:
            phi_loc = torch.matmul(
                eta, t_data[t, :].T
            )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

            phi_loc_centered = get_centered(phi_loc, p_types)

            phi = pyro.sample("phi", dist.Normal(phi_loc_centered, phi_scale))

        # story level

        with pyro.plate("s_indep", num_s_indeps, dim=-1):
            beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale))

        with pyro.plate("story", num_stories, dim=-1) as s:
            theta_loc = torch.matmul(
                beta, s_data[s, :].T
            )  # (num_p_indeps, num_s_indeps) x (num_s_indeps, num_stories)

            theta_loc_centered = get_centered(theta_loc, p_stories)

            theta = pyro.sample(
                "theta", dist.Normal(theta_loc_centered, theta_scale)
            )

        # subreddit level

        with pyro.plate("r_indep", num_r_indeps, dim=-1):
            tau = pyro.sample("tau", dist.Normal(tau_loc, tau_scale))

        with pyro.plate("subreddit", num_subreddits, dim=-1) as r:
            rho_loc = torch.matmul(
                tau, r_data[r, :].T
            )  # (num_p_indeps, num_r_indeps) x (num_r_indeps, num_subreddits)

            rho_loc_centered = get_centered(rho_loc, p_subreddits)

            rho = pyro.sample("rho", dist.Normal(rho_loc_centered, rho_scale))

        # shared
        gamma = pyro.sample("gamma", dist.Normal(gamma_loc, gamma_scale))

    # Gate

    gate_alpha = pyro.param(
        "gate_alpha",
        2.0 * torch.ones((num_types,), dtype=torch.float64),
        constraint=constraints.positive,
    )
    gate_beta = pyro.param(
        "gate_beta",
        2.0 * torch.ones((num_types,), dtype=torch.float64),
        constraint=constraints.positive,
    )
    with pyro.plate("type2", num_types, dim=-1):
        gate = pyro.sample("gate", dist.Beta(gate_alpha, gate_beta))

    return eta, phi, beta, theta, tau, rho, gate, gamma


def get_y_pred(
    p_data, t_data, s_data, r_data, p_types, p_stories, p_subreddits
):
    eta_loc = pyro.param("eta_loc").detach()
    beta_loc = pyro.param("beta_loc").detach()
    tau_loc = pyro.param("tau_loc").detach()
    gamma = pyro.param("gamma_loc").detach()

    phi = torch.matmul(eta_loc, t_data.T)
    theta = torch.matmul(beta_loc, s_data.T)
    rho = torch.matmul(tau_loc, r_data.T)

    t = torch.Tensor(p_types).long()
    s = torch.Tensor(p_stories).long()
    r = torch.Tensor(p_subreddits).long()

    phi_centered = get_centered(phi, t)
    theta_centered = get_centered(theta, s)
    rho_centered = get_centered(rho, r)

    indeps = torch.tensor(p_data)

    num_posts = p_data.shape[0]

    t_coefs = torch.tensor(phi_centered[:, t])  # (num_p_indeps,num_posts)
    s_coefs = torch.tensor(theta_centered[:, s])  # (num_p_indeps,num_posts)
    r_coefs = torch.tensor(rho_centered[:, r])  # (num_p_indeps,num_posts)
    shared_coefs = torch.tensor(gamma).repeat(
        (1, num_posts)
    )  # (num_p_indeps,num_posts)

    mu = (
        torch.mul(t_coefs, indeps.T)
        + torch.mul(s_coefs, indeps.T)
        + torch.mul(r_coefs, indeps.T)
        + torch.mul(shared_coefs, indeps.T)
    ).sum(dim=0)

    y_pred = np.exp(mu)

    return y_pred


def get_type_only_y_pred(p_data, t_data, s_data, r_data, p_types):
    eta_loc = pyro.param("eta_loc").detach()
    gamma = pyro.param("gamma_loc").detach()

    phi = torch.matmul(eta_loc, t_data.T)

    t = torch.Tensor(p_types).long()

    phi_centered = get_centered(phi, t)

    indeps = torch.tensor(p_data)

    num_posts = p_data.shape[0]

    t_coefs = torch.tensor(phi_centered[:, t])  # (num_p_indeps,num_posts)
    shared_coefs = torch.tensor(gamma).repeat(
        (1, num_posts)
    )  # (num_p_indeps,num_posts)

    mu = (
        torch.mul(t_coefs, indeps.T) + torch.mul(shared_coefs, indeps.T)
    ).sum(dim=0)

    y_pred = np.exp(mu)

    return y_pred
