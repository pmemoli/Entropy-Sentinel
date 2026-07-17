import torch


def compute_entropy_profile(logprobs_list):
    """The entropy profile is the shannon entropies of the token distributions at each step in the generation."""
    entropy_values = []

    for token_step_logprobs in logprobs_list:
        logprobs = torch.tensor(
            [v.logprob for v in token_step_logprobs.values()]
        )
        probs = torch.exp(logprobs)

        token_entropy = -torch.sum(probs * logprobs)
        entropy_values.append(token_entropy.item())

    return torch.tensor(entropy_values)


def compute_features(profile, selected_logprobs):
    """
    Input:
        profile: Tensor of shape [sequence_length]
        selected_logprobs: Tensor of shape [sequence_length]
    Output:
        metrics: Tensor of shape [num_statistical_summaries]
    """

    # Distribution summaries
    mean = torch.mean(profile)
    std = torch.std(profile)
    max = torch.max(profile)

    q10 = torch.quantile(profile, 0.10)
    q25 = torch.quantile(profile, 0.25)
    q50 = torch.quantile(profile, 0.50)
    q75 = torch.quantile(profile, 0.75)
    q90 = torch.quantile(profile, 0.90)

    centered = profile - mean
    skewness = torch.mean(centered**3) / (std**3 + 1e-8)
    kurtosis = torch.mean(centered**4) / (std**4 + 1e-8)

    # Baseline UQ metrics
    se_sum = torch.sum(profile)
    nll_sum = torch.sum(-selected_logprobs)
    nll_avg = torch.mean(-selected_logprobs)
    nll_max = torch.max(-selected_logprobs)
    lntp = torch.sum(-selected_logprobs * torch.log(-selected_logprobs + 1e-8))
    mtp = torch.max(-selected_logprobs * torch.log(-selected_logprobs + 1e-8))
    ppl = torch.exp(nll_avg)

    features = torch.tensor(
        [
            mean,
            std,
            max,
            q10,
            q25,
            q50,
            q75,
            q90,
            skewness,
            kurtosis,
            se_sum,
            nll_avg,
            nll_max,
            nll_sum,
            lntp,
            mtp,
            ppl,
        ],
    )

    return features
