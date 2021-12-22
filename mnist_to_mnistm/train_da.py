import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm
import time
import os
import json
from pathlib import Path
import logging

from da.base import DABase
from models import Net
from datasets import get_mnist, get_mnistm


def main(run_conf, da_conf, exp_dir):
    clf_model = Net().to(run_conf['device'])
    print("Clf model: ", clf_model)

    # get da base object from da package
    if da_conf['da_type']:
        embeds_size = clf_model.get_embed_size()
        da = DABase(embeds_size, **da_conf)
    else:
        da = None

    # load datasets used to test da package
    mnist_train_ds = get_mnist('data', train=True,
                               transform=Compose([ToTensor(),
                                                  transforms.Normalize((0.5,), (0.5,))]))
    mnistm_train_ds = get_mnistm('data', train=True,
                                 transform=Compose([ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    mnist_test_ds = get_mnist('data', train=False, transform=Compose([ToTensor(),
                                                                      transforms.Normalize((0.5,), (0.5,))]))
    mnistm_test_ds = get_mnistm('data', train=False, transform=Compose([ToTensor(),
                                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    # save images for visual inspection (normalized ones!)
    img_folder = os.path.join(exp_folder, 'images')
    os.makedirs(img_folder, exist_ok=True)
    print("Saving data sample images to: ", img_folder)
    save_image(mnist_train_ds[0][0], os.path.join(img_folder, 'src_train.png'))
    save_image(mnistm_train_ds[0][0], os.path.join(img_folder, 'tgt_train.png'))
    save_image(mnist_test_ds[0][0], os.path.join(img_folder, 'src_test.png'))
    save_image(mnistm_test_ds[0][0], os.path.join(img_folder, 'tgt_test.png'))

    # create dataloaders
    half_batch = run_conf['batch_size'] // 2
    src_train_dl = DataLoader(mnist_train_ds, batch_size=half_batch, shuffle=True)
    tgt_train_dl = DataLoader(mnistm_train_ds, batch_size=half_batch, shuffle=True)
    src_test_dl = DataLoader(mnist_test_ds, batch_size=half_batch, shuffle=True)
    tgt_test_dl = DataLoader(mnistm_test_ds, batch_size=half_batch, shuffle=True)

    if da_config['da_type'] == "dann" and not da_config['da_spec_config']['dann']['auto_critic_update']:
        # in case of dann da net parameters are most commonly optimized outside package (grl)
        clf_optim = torch.optim.Adam(list(clf_model.parameters()) + da.get_da_params(), lr=run_conf['lr'],
                                     weight_decay=run_conf['weight_decay'])
    else:
        clf_optim = torch.optim.Adam(clf_model.parameters(), lr=run_conf['lr'], weight_decay=run_conf['weight_decay'])
    clf_criterion = nn.CrossEntropyLoss()

    model_file = os.path.join(exp_dir, 'model.pt')
    if Path(model_file).exists():
        print("Loading model and skipping training, proceed with testing: ")
        clf_model.load_state_dict(torch.load(model_file))
        test(src_test_dl, tgt_test_dl, clf_model, run_conf['device'])
    else:
        for epoch in range(1, run_conf['epochs'] + 1):
            clf_loss, da_loss = train_step(src_train_dl, tgt_train_dl, da, clf_model,
                                           clf_optim, clf_criterion, run_conf['device'])
            logger.info(f'EPOCH {epoch:03d}: clf_loss={clf_loss:.4f}, da_loss={da_loss:.4f}, '
                        f'lambda={da.get_current_lambda() if da else 0.0:.4f}')
            test(src_test_dl, tgt_test_dl, clf_model, run_conf['device'])
        # save trained model
        torch.save(clf_model.state_dict(), model_file)
        print("Saved model to: ", model_file)


def train_step(src_train_dl, tgt_train_dl, da, model, optimizer, criterion, device):
    model.train()
    total_clf_loss = 0
    total_da_loss = 0
    for (src_x, src_y), (tgt_x, _) in tqdm(zip(src_train_dl, tgt_train_dl), total=len(src_train_dl)):
        # expand channel dimension of mnist to match mnistm
        src_x = src_x.expand(src_x.size(0), 3, src_x.size(2), src_x.size(3))
        x = torch.cat((src_x, tgt_x)).to(device)
        # use dummy labels for tgt samples
        y = torch.cat((src_y, torch.zeros(len(tgt_x), dtype=torch.int64))).to(device)
        dl = torch.cat((torch.zeros(len(src_x), dtype=torch.int64),
                        torch.ones(len(tgt_x), dtype=torch.int64))).to(device)
        embeds, y_hat = model(x)

        # cross-entropy loss based on source data
        clf_loss = criterion(y_hat[:len(src_y)], y[:len(src_y)])

        if da:
            # let da loss be calculated by da package
            da_loss, da_info = da.get_da_loss(embeds, dl, y, y_hat)
        else:
            da_loss = torch.tensor(0., device=device)

        loss = clf_loss + da_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_clf_loss += clf_loss.item()
        total_da_loss += da_loss.item()
    mean_clf_loss = total_clf_loss / len(src_train_dl)
    mean_da_loss = total_da_loss / len(src_train_dl)
    return mean_clf_loss, mean_da_loss


def test(src_test_dl, tgt_test_dl, model, device):
    model.eval()

    def run(dl, model_, device_):
        preds = []
        for x, y in dl:
            x = x.expand(x.size(0), 3, x.size(2), x.size(3))
            x = x.to(device_)
            y = y.to(device_)
            with torch.no_grad():
                _, y_hat = model_(x)
            preds.append(y_hat.argmax(dim=1) == y)
        preds = torch.cat(preds).detach().cpu()
        return torch.true_divide(torch.sum(preds), len(preds))

    logger.info(f'Test Accuracy on MNIST: {run(src_test_dl, model, device)}')
    logger.info(f'Test Accuracy on MNISTM: {run(tgt_test_dl, model, device)}')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation')
    arg_parser.add_argument('--da_type', type=str, default=None)
    arg_parser.add_argument('--batch_size', type=int, default=64)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--weight_decay', type=float, default=1e-3)
    arg_parser.add_argument('--da_lambda', type=float, default=1.0)
    arg_parser.add_argument('--epochs', type=int, default=50)
    arg_parser.add_argument('--device', type=str, default='cuda:0')
    arg_parser.add_argument('--data_dir', type=str, default='data')
    arg_parser.add_argument('--exp_dir', type=str, default=None)
    arg_parser.add_argument('--all_embeds', action='store_true', default=False)

    args = arg_parser.parse_args()

    if args.exp_dir:
        exp_folder = args.exp_dir
    else:
        # create experiment folder
        timestr = str(args.da_type) + time.strftime("%Y%m%d-%H%M%S")
        exp_folder = os.path.join('experiments', timestr)
        os.makedirs(exp_folder)
        print("Created experiment folder: ", exp_folder)

    # logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('train_da')
    file_log_handler = logging.FileHandler(os.path.join(exp_folder, 'exp_out.log'))
    logger.addHandler(file_log_handler)

    run_config = {
        'data_dir': args.data_dir,
        'device': args.device if torch.cuda.is_available() else 'cpu',
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs
    }

    da_config = {
        'embeds_idx': [0, 1, 2, 3] if args.all_embeds else [1],
        'da_type': args.da_type,
        'da_lambda': args.da_lambda,
        'lambda_auto_schedule': True,
        'lambda_pretrain_steps': 6000,
        'lambda_inc_steps': 70000,
        'lambda_final': args.da_lambda,
        'num_domains': 2,
        'num_classes': 10,
        # adversarial net parameters
        'adv_config': {
            'da_net_config': {
                'layers_width': [
                    100
                ],
                'act_function': "relu",
                'dropout': 0.0
            },
            'da_optimizer_config': {
                'lr': 0.0001,
                'weight_decay': 0.001
            }
        },
        # da algorithm specific parameters
        'da_spec_config':
            {
                'cmd': {
                    'n_moments': 5
                },
                'coral': {},
                'dann': {
                    'grad_scale_factor': 1.0,
                    'auto_critic_update': False,
                    # ignored if auto_critic_update is False
                    'critic_iter': 5,
                    'da_optimizer_config': {
                        'lr': 0.0001,
                        'weight_decay': 0.001
                    }
                },
                'jdot': {
                    'jdot_alpha': 1.0
                },
                'mmd': {
                    'kernel_mul': 2,
                    'kernel_num': 4,
                    'fix_sigma': None
                },
                'swd': {
                    'multiplier': 8,
                    'p': 2
                },
                'wdgrl': {
                    'gp_da_lambda': 1.0,
                    'critic_iter': 5,
                },
            }
    }

    # save settings of the experiment
    with open(os.path.join(exp_folder, 'run_config.json'), 'w+') as f:
        json.dump(run_config, f)

    with open(os.path.join(exp_folder, 'da_config.json'), 'w+') as f:
        json.dump(da_config, f)

    main(run_config, da_config, exp_folder)
