import torch
import torch.nn.functional as F

from da.da_helpers import RevGrad


def dann_loss(embeds, domain_labels, grad_scale_factor, da_net, da_info):
    loss = torch.tensor(0., device=domain_labels.device)
    da_info['embed_accuracies'] = []
    for i in range(len(embeds)):
        # RevGrad is static - only one grad_scale_factor possible for all embeddings
        embeds[i] = RevGrad.apply(embeds[i], grad_scale_factor)
    # apply adversarial da networks
    da_preds = da_net.forward(embeds)
    for da_pred in da_preds:
        embed_loss = F.cross_entropy(da_pred, domain_labels)
        embed_acc = (da_pred.max(dim=1)[1] == domain_labels).float().sum() / len(domain_labels)
        loss += embed_loss
        da_info['embed_losses'].append(embed_loss.detach().cpu())
        da_info['embed_accuracies'].append(embed_acc.detach().cpu())
    return loss


def dann_update(embeds, domain_labels, da_optimizer, critic_iter, da_net):
    # iterate through all embeddings and the according da_nets
    for i, embed in enumerate(embeds):
        # multiple iterations of training the critic
        for _ in range(critic_iter):
            # feature extractor is not updated here - detach embeddings
            critic_out = da_net.nets[i](embed.detach())
            loss = F.cross_entropy(critic_out, domain_labels)
            da_optimizer.zero_grad()
            loss.backward()
            da_optimizer.step()
