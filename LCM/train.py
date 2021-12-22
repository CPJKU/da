import torch
from tqdm import tqdm


# Train function called in every epoch
def train(model, da_model, data_loader, reg_loss, optim, device):
    """

    Parameters
    ----------
    model: Pytorch Model
    da_model : domain adaptation model created in the train da file
    data_loader: train dataloder
    reg_loss: loss criterion, MSE if regression
    optim: optimizer of the network
    device: device that network and parameters are in

    Returns
    -------
    Domain Loss
    Regression Loss
    """
    model.train()
    total_clf_loss, total_da_loss = 0, 0
    mean_clf_loss, mean_da_loss = 0, 0
    for signal, domain_label, phi_offset in tqdm(data_loader):
        signal, domain_label, phi_offset = signal.to(device), domain_label.to(device), phi_offset.to(device)
        embeds, preds = model(signal)
        source_embeds, source_preds = embeds[domain_label == 0], preds[domain_label == 0]
        target_embeds = embeds[domain_label == 1]
        clf_loss = reg_loss(preds, phi_offset)
        if da_model is not None:
            # prepare data for da object
            da_loss, da_info = da_model.get_da_loss([torch.cat((source_embeds, target_embeds))], domain_label)
            total_da_loss += da_loss.item()
        else:
            da_loss = 0

        total_clf_loss += clf_loss.item()
        loss = clf_loss + da_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        mean_clf_loss = total_clf_loss / len(data_loader)
        mean_da_loss = total_da_loss / len(data_loader)
    return mean_clf_loss, mean_da_loss


def evaluate(model, data_loader, loss, device):
    """

    Parameters
    ----------
    model: Pytorch Model
    data_loader: usually Validation and test dataloaders
    loss: loss criterion for the validation can be any custom loss function e.g accuracy
    device: device that parameters should be in
    Returns
    -------

    """
    total_clf_loss = 0

    model.eval()
    pred_list = []
    targets = []
    for signal, domain_label, phi_offset in data_loader:
        signal, domain_label, phi_offset = signal.to(device), domain_label.to(device), phi_offset.to(device)
        with torch.no_grad():
            _, preds = model(signal)
        pred_list.append(preds)
        targets.append(phi_offset)
        clf_loss = loss(preds, phi_offset)
        total_clf_loss += clf_loss.item()
    pred_list = torch.cat(pred_list).detach().cpu()
    targets = torch.cat(targets).detach().cpu()
    mean_clf_loss = total_clf_loss / len(data_loader)
    return mean_clf_loss, mae(targets, pred_list), rmse(targets, pred_list)


def mae(y, y_pred):
    return torch.mean(torch.abs(y-y_pred))


def rmse(y, y_pred):
    return torch.sqrt(torch.mean((y-y_pred)**2))


# wrapping training and evaluation function for the number of epochs
def train_model(model, da_model, train_loader, test_loader, loss, optim, epochs, device, logger):
    test_loss, test_mae, test_rmse = evaluate(model, test_loader, loss, device)
    logger.info('Evaluation Loss at the beginning of the Training: %0.4f' % test_loss)
    for i in range(epochs):
        train_loss, da_loss = train(model, da_model, train_loader, loss, optim, device)
        test_loss, test_mae, test_rmse = evaluate(model, test_loader, loss, device)
        logger.info(f'EPOCH {(i+1):03d}: clf_loss={train_loss:.4f}, da_loss={da_loss:.4f}, test_loss ={test_loss:.4f},'
                    f'test_mae ={test_mae:.4f}, test_rmse ={test_rmse:.4f}')
