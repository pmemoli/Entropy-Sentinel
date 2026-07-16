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
