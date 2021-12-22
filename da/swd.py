import torch
import math


def swd_loss(embeds, domain_labels, multiplier, p, da_info):
    loss = torch.tensor(0., device=domain_labels.device)
    # split into source and target samples
    unique_dl = domain_labels.unique()
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return loss
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # for all embeds calculate the mmd loss and sum them together
    for embed in embeds:
        src_embed = embed[src_mask]
        tgt_embed = embed[tgt_mask]
        embed_loss = swd(src_embed, tgt_embed, multiplier, p)
        da_info['embed_losses'].append(embed_loss.detach().cpu())
        loss += embed_loss
    return loss


def swd(src_embed, tgt_embed, multiplier, p):
    projections = torch.zeros((src_embed.size(1), src_embed.size(1) * multiplier),
                              device=tgt_embed.device).normal_(0, 1)
    projections = projections / torch.norm(projections, p=p, dim=0, keepdim=True)

    # repeat target batch size to be the same size as source
    src_batch_size = src_embed.size(0)
    tgt_batch_size = tgt_embed.size(0)
    batch_size = src_batch_size + tgt_batch_size
    src_repeats = math.ceil(batch_size / src_batch_size)
    tgt_repeats = math.ceil(batch_size / tgt_batch_size)
    src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
    tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]

    # project both samples 'num_projections' times
    pr_src = src_embed_rep.mm(projections)
    pr_tgt = tgt_embed_rep.mm(projections)

    # sort the projection results
    pr_sim = torch.sort(pr_src, dim=0)[0]
    pr_meas = torch.sort(pr_tgt, dim=0)[0]
    sliced_wd = torch.pow(pr_sim - pr_meas, p)

    # return mean distance, scaled by batch size
    return sliced_wd.mean()
