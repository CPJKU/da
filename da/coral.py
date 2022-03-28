import torch
import math


def coral_loss(embeds, domain_labels, da_info):
    loss = torch.tensor(0., device=domain_labels.device)
    # split into source and target samples
    unique_dl = domain_labels.unique()
    # if a batch with samples of only one domain is encountered - return 0 as loss
    if len(unique_dl) == 1:
        return loss
    src_mask = domain_labels == unique_dl[0]
    tgt_mask = domain_labels == unique_dl[1]

    # for all embeds calculate the coral loss and sum them together
    for embed in embeds:
        src_embed = embed[src_mask]
        tgt_embed = embed[tgt_mask]
        embed_loss = coral(src_embed, tgt_embed)
        da_info['embed_losses'].append(embed_loss.detach().cpu())
        loss += embed_loss
    return loss


def comp_cov(x):
    xm = x - torch.mean(x, dim=0, keepdim=True)
    return xm.T @ xm / (x.size(0) - 1)


def coral(src_embed, tgt_embed):
    src_batch_size = src_embed.size(0)
    tgt_batch_size = tgt_embed.size(0)
    batch_size = src_batch_size + tgt_batch_size
    src_repeats = math.ceil(batch_size / src_batch_size)
    tgt_repeats = math.ceil(batch_size / tgt_batch_size)
    # handle case when source and target are not of same size
    src_embed_rep = torch.cat([src_embed] * src_repeats, dim=0)[:batch_size]
    tgt_embed_rep = torch.cat([tgt_embed] * tgt_repeats, dim=0)[:batch_size]
    d = src_embed_rep.size()[1]
    src_cov = comp_cov(src_embed_rep)
    tgt_cov = comp_cov(tgt_embed_rep)

    # squared matrix frobenius norm
    loss = torch.sum((src_cov - tgt_cov)**2)
    loss = loss / (4 * d * d)
    return loss
