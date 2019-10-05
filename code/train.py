from config import *
import datetime
import json
import os
from loader import get_loaders, split_train_val
from torch.utils.data import DataLoader
from model import get_model
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

torch.manual_seed(1)


def bnwd_optim_params(model, model_params):
    bn_params, remaining_params = split_bn_params(model, model_params)
    return [{'params': bn_params, 'weight_decay': 0}, {'params': remaining_params}]


def split_bn_params(model, model_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)

    bn_params = [p for p in model_params if p in mod_bn_params]
    rem_params = [p for p in model_params if p not in mod_bn_params]
    return bn_params, rem_params


EPSILON = 1e-7


def train_model(cfg: Config, weight_path=None, device='cuda:0'):
    now = datetime.datetime.now()
    log_dir = os.path.join(MODEL_DIR, f'{cfg.NAME.lower()}_{now:%Y%m%dT%H%M}')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=5)

    # snapshot
    with open(os.path.join(log_dir, 'snapshot.txt'), 'w') as f:
        snapshot = cfg.get_snapshot()
        json.dump(snapshot, f, indent=4)

    model = get_model(cfg)
    model.to(device)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))

    if cfg.LOSS == 'ce':
        weight = None
        criterion = nn.CrossEntropyLoss(weight=weight)
    elif cfg.LOSS == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.5, dtype=torch.float))
    elif cfg.LOSS == 'focal_loss':
        criterion = BCEFocalLoss()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if cfg.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=cfg.BASE_LR, momentum=0.9, weight_decay=5e-4)
    elif cfg.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=cfg.BASE_LR, weight_decay=5e-4)

    if cfg.SCHEDULER == 'step':
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    elif cfg.SCHEDULER == 'multstep':
        scheduler = MultiStepLR(optimizer, milestones=(20, 40), gamma=0.1)

    gloabl_step = 0
    train_loader, val_loader = get_loaders(cfg)

    for epoch in range(1, cfg.EPOCHS + 1):
        batch_loss = []
        train_loss = []

        model.train()

        # scheduler(optimizer, epoch)

        scheduler.step(epoch)

        pr, gt = [], []
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            pr.extend(torch.round(torch.sigmoid(outputs)).detach().cpu().numpy().squeeze())
            gt.extend(labels.cpu().numpy().squeeze())

            # print(pr)
            # print(gt)

            batch_loss.append(loss.item())
            loss = loss / cfg.ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % cfg.ACCUMULATION_STEPS == 0:
                # scheduler.step()
                # adjust_learning_rate(optimizer, cfg.BASE_LR, gloabl_step, epoch,
                #                      warmup_iters=len(train_loader) // (cfg.ACCUMULATION_STEPS * cfg.IMAGE_PER_GPU) * 5)
                optimizer.step()
                optimizer.zero_grad()
                gloabl_step += 1

                train_loss.append(np.mean(batch_loss))
                batch_loss = np.mean(batch_loss)

                lr = optimizer.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('lr', lr,
                                  gloabl_step)

                print(
                    f'epoch {epoch:5d} batch {(i + 1) // cfg.ACCUMULATION_STEPS:5d}, loss:{np.mean(batch_loss):.4f}, lr:{lr:.4e}')

                writer.add_scalar('batch_loss', np.mean(batch_loss), gloabl_step)
                batch_loss = []
            # break
        train_acc = accuracy_score(gt, pr)
        train_recall = recall_score(gt, pr)
        train_precision = precision_score(gt, pr)
        print(confusion_matrix(gt, pr))
        print(
            f'epoch {epoch} mean_loss:{np.mean(train_loss):.4f} acc:{train_acc:.4f} recall:{train_recall:.4f} '
            f'precision:{train_precision:.4f} pos_num:{sum(gt)} neg_num:{len(gt) - sum(gt)}')

        model.eval()
        val_loss = []

        pr, gt = [], []
        with torch.no_grad():
            for j, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

                pr.extend(torch.round(torch.sigmoid(outputs)).detach().cpu().numpy().squeeze())
                gt.extend(labels.cpu().numpy().squeeze())

        val_acc = accuracy_score(gt, pr)
        val_recall = recall_score(gt, pr)
        val_precision = precision_score(gt, pr)
        print(confusion_matrix(gt, pr))

        print(
            f'epoch {epoch} val_loss:{np.mean(val_loss):.4f} acc:{val_acc:.4f} recall:{val_recall:.4f} '
            f'precision:{val_precision:.4f} pos_num:{sum(gt)} neg_num:{len(gt) - sum(gt)}')

        checkpoint_path = os.path.join(log_dir,
                                       "{}_{:04d}_{:.4f}.pth".format(cfg.NAME.lower(), epoch, np.mean(val_acc)))

        writer.add_scalars('loss', {'loss': np.mean(train_loss), 'val_loss': np.mean(val_loss)}, epoch)

        writer.add_scalars('acc',
                           {'train_acc': train_acc, 'train_precsion': train_precision, 'train_reall': train_recall},
                           epoch)
        writer.add_scalars('val_acc',
                           {'val_acc': val_acc, 'val_precsion': val_precision, 'val_reall': val_recall}, epoch)

        torch.save(model.state_dict(), checkpoint_path)

    writer.close()


if __name__ == '__main__':
    import socket

    if socket.gethostname() == 'Wayne':
        device = torch.device('cuda:0')
        cfg = Config()
        # cfg = UnetP1Crop()
        # cfg = UnetP1()
        # cfg = P3Cls()
        # cfg = UnetP1Crop()
        # cfg = P1Cls()
        # cfg = UnetP1CropW()

        weight_path = None
        # cfg = CHECK()
    elif socket.gethostname() == 'gaps':
        device = torch.device('cuda:0')

        weight_path = None

    train_model(cfg, weight_path)
