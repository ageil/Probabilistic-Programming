import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints


def complete_model(
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

    # subreddit priors
    tau_loc = torch.zeros((num_p_indeps, num_r_indeps), dtype=torch.float64)
    tau_scale = coef_scale_prior * torch.ones(
        (num_p_indeps, num_r_indeps), dtype=torch.float64
    )

    if zero_inflated:
        # type priors
        eta_gate_loc = torch.zeros(
            (num_p_indeps, num_t_indeps), dtype=torch.float64
        )
        eta_gate_scale = coef_scale_prior * torch.ones(
            (num_p_indeps, num_t_indeps), dtype=torch.float64
        )

        # story priors
        beta_gate_loc = torch.zeros(
            (num_p_indeps, num_s_indeps), dtype=torch.float64
        )
        beta_gate_scale = coef_scale_prior * torch.ones(
            (num_p_indeps, num_s_indeps), dtype=torch.float64
        )

        # subreddit priors
        tau_gate_loc = torch.zeros(
            (num_p_indeps, num_r_indeps), dtype=torch.float64
        )
        tau_gate_scale = coef_scale_prior * torch.ones(
            (num_p_indeps, num_r_indeps), dtype=torch.float64
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
            phi_loc = torch.matmul(eta, t_data[t, :].T)
            # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

            phi = pyro.sample("phi", dist.Normal(phi_loc, coef_scale_prior))

            if zero_inflated:
                phi_gate_loc = torch.matmul(
                    eta_gate, t_data[t, :].T
                )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

                phi_gate = pyro.sample(
                    "phi_gate", dist.Normal(phi_gate_loc, coef_scale_prior)
                )

        # Story Level
        with pyro.plate("s_indep", num_s_indeps, dim=-1):
            beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale))
            if zero_inflated:
                beta_gate = pyro.sample(
                    "beta_gate", dist.Normal(beta_gate_loc, beta_gate_scale)
                )

        with pyro.plate("story", num_stories, dim=-1) as s:
            theta_loc = torch.matmul(
                beta, s_data[s, :].T
            )  # (num_p_indeps, num_s_indeps) x (num_s_indeps, num_stories)

            theta = pyro.sample(
                "theta", dist.Normal(theta_loc, coef_scale_prior)
            )

            if zero_inflated:
                theta_gate_loc = torch.matmul(
                    beta_gate, s_data[s, :].T
                )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

                theta_gate = pyro.sample(
                    "theta_gate", dist.Normal(theta_gate_loc, coef_scale_prior)
                )

        # Subreddit Level
        with pyro.plate("r_indep", num_r_indeps, dim=-1):
            tau = pyro.sample("tau", dist.Normal(tau_loc, tau_scale))
            if zero_inflated:
                tau_gate = pyro.sample(
                    "tau_gate", dist.Normal(tau_gate_loc, tau_gate_scale)
                )

        with pyro.plate("subreddit", num_subreddits, dim=-1) as r:
            rho_loc = torch.matmul(
                tau, r_data[r, :].T
            )  # (num_p_indeps, num_r_indeps) x (num_r_indeps, num_subreddits)

            rho = pyro.sample("rho", dist.Normal(rho_loc, coef_scale_prior))
            if zero_inflated:
                rho_gate_loc = torch.matmul(
                    tau_gate, r_data[r, :].T
                )  # (num_p_indeps, num_t_indeps) x (num_t_indeps, num_types)

                rho_gate = pyro.sample(
                    "rho_gate", dist.Normal(rho_gate_loc, coef_scale_prior)
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

        type_level_products = torch.mul(
            t_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
        story_level_products = torch.mul(
            s_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
        subreddit_level_products = torch.mul(
            r_coefs, indeps.T
        )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

        # calculate the mean: desired shape (num_posts, 1)
        mu = (
            subreddit_level_products
            + type_level_products
            + story_level_products
        ).sum(
            dim=0
        )  # (num_p_indeps, num_posts).sum(over indeps)

        # defining response dist
        if zero_inflated:
            t_coefs_gate = phi_gate[:, t]  # (num_p_indeps,num_posts)
            s_coefs_gate = theta_gate[:, s]  # (num_p_indeps,num_posts)
            r_coefs_gate = rho_gate[:, r]  # (num_p_indeps,num_posts)

            type_level_products_gate = torch.mul(
                t_coefs_gate, indeps.T
            )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
            story_level_products_gate = torch.mul(
                s_coefs_gate, indeps.T
            )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)
            subreddit_level_products_gate = torch.mul(
                r_coefs_gate, indeps.T
            )  # (num_p_indeps, num_posts) .* (num_p_indeps, num_posts)

            # calculate the mean: desired shape (num_posts, 1)
            gate = torch.nn.Sigmoid()(
                (
                    type_level_products_gate
                    + story_level_products_gate
                    + subreddit_level_products_gate
                ).sum(dim=0)
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


def complete_guide(
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
    num_stories, num_s_indeps = s_data.shape
    num_subreddits, num_r_indeps = r_data.shape

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

        # story level:
        beta_gate_loc = pyro.param(
            "beta_gate_loc",
            torch.zeros((num_p_indeps, num_s_indeps), dtype=torch.float64),
        )
        beta_gate_scale = pyro.param(
            "beta_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, num_s_indeps), dtype=torch.float64),
            constraint=constraints.positive,
        )

        theta_gate_scale = pyro.param(
            "theta_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, 1), dtype=torch.float64),
            constraint=constraints.positive,
        )  # share among all types.

        # subreddit level:
        tau_gate_loc = pyro.param(
            "tau_gate_loc",
            torch.zeros((num_p_indeps, num_r_indeps), dtype=torch.float64),
        )
        tau_gate_scale = pyro.param(
            "tau_gate_scale",
            coef_scale_prior
            * torch.ones((num_p_indeps, num_r_indeps), dtype=torch.float64),
            constraint=constraints.positive,
        )

        rho_gate_scale = pyro.param(
            "rho_gate_scale",
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

        # story level
        with pyro.plate("s_indep", num_s_indeps, dim=-1):
            beta = pyro.sample("beta", dist.Normal(beta_loc, beta_scale))
            if zero_inflated:
                beta_gate = pyro.sample(
                    "beta_gate", dist.Normal(beta_gate_loc, beta_gate_scale)
                )

        with pyro.plate("story", num_stories, dim=-1) as s:
            theta_loc = torch.matmul(
                beta, s_data[s, :].T
            )  # (num_p_indeps, num_s_indeps) x (num_s_indeps, num_stories)

            pyro.sample("theta", dist.Normal(theta_loc, theta_scale))

            if zero_inflated:
                theta_gate_loc = torch.matmul(
                    beta_gate, s_data[s, :].T
                )  # (num_p_indeps, num_s_indeps) x (num_s_indeps, num_stories)

                pyro.sample(
                    "theta_gate", dist.Normal(theta_gate_loc, theta_gate_scale)
                )

        # subreddit level
        with pyro.plate("r_indep", num_r_indeps, dim=-1):
            tau = pyro.sample("tau", dist.Normal(tau_loc, tau_scale))
            if zero_inflated:
                tau_gate = pyro.sample(
                    "tau_gate", dist.Normal(tau_gate_loc, tau_gate_scale)
                )

        with pyro.plate("subreddit", num_subreddits, dim=-1) as r:
            rho_loc = torch.matmul(
                tau, r_data[r, :].T
            )  # (num_p_indeps, num_r_indeps) x (num_r_indeps, num_subreddits)

            pyro.sample("rho", dist.Normal(rho_loc, rho_scale))

            if zero_inflated:
                rho_gate_loc = torch.matmul(tau_gate, r_data[r, :].T)
                # (num_p_indeps, num_r_indeps) x (num_r_indeps, num_subreddits)
                pyro.sample(
                    "rho_gate", dist.Normal(rho_gate_loc, rho_gate_scale)
                )
