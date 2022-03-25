import torch


# code by authors of cmd paper:
# https://github.com/wzell/cmd


def cmd_loss(embeds, domain_labels, n_moments, da_info):
    loss = torch.tensor(0., device=domain_labels.device)
    # split into source and target samples
    unique_dl = domain_labels.unique(sorted=True)
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return loss
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # sum up cmd loss for every pair of embeddings
    for embed in embeds:
        src_embed = embed[src_mask]
        tgt_embed = embed[tgt_mask]

        embed_loss = cmd(src_embed, tgt_embed, n_moments)
        da_info['embed_losses'].append(embed_loss.detach().cpu())

        loss += embed_loss
    return loss


def cmd(src_embed, tgt_embed, n_moments):
    if torch.mean(torch.abs(src_embed) + torch.abs(tgt_embed)) <= 1e-7:
        print("Warning: feature representations tend towards zero. "
              "Consider decreasing 'da_lambda' or using lambda schedule.")

    src_mean = src_embed.mean(dim=0)
    tgt_mean = tgt_embed.mean(dim=0)

    src_centered = src_embed - src_mean
    tgt_centered = tgt_embed - tgt_mean

    first_moment = l2diff(src_mean, tgt_mean)  # start with first moment

    moments_diff_sum = first_moment
    for k in range(2, n_moments + 1):
        moments_diff_sum = moments_diff_sum + moment_diff(src_centered, tgt_centered, k)

    return moments_diff_sum


def l2diff(src, tgt):
    """
    standard euclidean norm. small number added to increase numerical stability.
    """
    return torch.sqrt(torch.sum((src - tgt) ** 2) + 1e-8)


def moment_diff(src, tgt, moment):
    """
    difference between moments
    """
    ss1 = (src ** moment).mean(0)
    ss2 = (tgt ** moment).mean(0)
    return l2diff(ss1, ss2)
