import numpy as np
import pyro
import torch


def get_y_pred(
    p_data,
    t_data=None,
    s_data=None,
    r_data=None,
    p_types=None,
    p_stories=None,
    p_subreddits=None,
):
    indeps = p_data
    total_coefs = torch.zeros((p_data.shape[1], p_data.shape[0]))
    global_param_names = [param_tup for param_tup in pyro.get_param_store()]

    higher_level_reg_params = ["eta_loc", "beta_loc", "tau_loc"]

    # add the coefficients coming from all groups together
    # (which params exist will differ depending on which model)
    for higher_level_reg_param_name in higher_level_reg_params:
        if higher_level_reg_param_name in global_param_names:
            if higher_level_reg_param_name == "eta_loc":
                group_data = t_data
                p_groups = p_types
            elif higher_level_reg_param_name == "beta_loc":
                group_data = s_data
                p_groups = p_stories
            elif higher_level_reg_param_name == "tau_loc":
                group_data = r_data
                p_groups = p_subreddits

            higher_level_reg_param = pyro.param(
                higher_level_reg_param_name
            ).detach()

            coef_matrix = torch.matmul(higher_level_reg_param, group_data.T)

            group_coefs = coef_matrix[:, p_groups]  # (num_p_indeps,num_posts)

            total_coefs += group_coefs

    if "gamma_loc" in global_param_names:
        total_coefs += pyro.param("gamma_loc").detach()

    mu = (torch.mul(total_coefs, indeps.T)).sum(dim=0)

    y_pred = np.exp(mu).int()

    return y_pred


# Note: use 0s for any means that are not relevant to the model.
def get_type_only_y_pred(p_data, t_data, p_types, s_means=0, r_means=0):
    eta_loc = pyro.param("eta_loc").detach()

    phi = torch.matmul(eta_loc, t_data.T)

    indeps = p_data

    total_coefs = phi[:, p_types]  # (num_p_indeps,num_posts)
    total_coefs += s_means + r_means

    mu = (torch.mul(total_coefs, indeps.T)).sum(dim=0)

    y_pred = np.exp(mu).int()

    return y_pred


# Note: use 0s for any means that are not relevant to the model.
def get_mean_y_pred(p_data, t_means=0, s_means=0, r_means=0):

    indeps = p_data

    total_coefs = torch.zeros((p_data.shape[1], p_data.shape[0]))
    total_coefs += t_means + s_means + r_means

    mu = (torch.mul(total_coefs, indeps.T)).sum(dim=0)

    y_pred = np.exp(mu).int()

    return y_pred


# returns a (num_p_indeps, 1) tensor of means w.r.t the groups.
def get_means(loc, p_groups):

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

    return loc_means


def get_s_means(p_stories, s_data):
    beta_loc = pyro.param("beta_loc").detach()
    theta_loc = torch.matmul(beta_loc, s_data.T)
    s_means = get_means(theta_loc, p_stories)
    return s_means


def get_r_means(p_subreddits, r_data):
    tau_loc = pyro.param("tau_loc").detach()
    rho_loc = torch.matmul(tau_loc, r_data.T)
    r_means = get_means(rho_loc, p_subreddits)
    return r_means


def get_t_means(p_types, t_data):
    eta_loc = pyro.param("eta_loc").detach()
    phi_loc = torch.matmul(eta_loc, t_data.T)
    t_means = get_means(phi_loc, p_types)
    return t_means
