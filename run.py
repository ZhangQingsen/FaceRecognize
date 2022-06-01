import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

import dataloader
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.Resize((114, 114)),
                                transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def save_model(_model, dir_path, epoch, _loss, accu):
    torch.save(_model.state_dict(), os.path.join(dir_path, 'ep%03d-loss%.3f-accu%.3f.pth' % ((epoch + 1), _loss, accu)))
    # 加载数据


def loss(y_pred, Batch_size, alpha=0.2):
    anchor, positive, negative = y_pred[:int(Batch_size)], y_pred[int(Batch_size):int(2 * Batch_size)], y_pred[
                                                                                                        int(2 * Batch_size):]

    pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
    neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

    keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
    hard_triplets = np.where(keep_all == 1)

    pos_dist = pos_dist[hard_triplets]
    neg_dist = neg_dist[hard_triplets]

    basic_loss = pos_dist - neg_dist + alpha
    res_loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))
    return res_loss


def run(train_df, test_df, curr_epoch, epoch_step, Batch_size, lr=0.01):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0

    train_loader = DataLoader(dataloader.DataLoader(train_df, 114), batch_size=129, shuffle=True)
    test_loader = DataLoader(dataloader.DataLoader(test_df, 114), batch_size=129, shuffle=True)

    model_train = model.ConvNet(num_classes=len(np.unique(train_df["Name"])))
    scaler = torch.cuda.amp.GradScaler()
    optimizer = optim.SGD(model_train.parameters(), lr, momentum=0.9)
    # optimizer = optim.Adam(model_train.parameters(), lr=0.01,betas=(0.9,0.999))
    for e in range(epoch_step):
        for iteration, batch in enumerate(train_loader):
            images, labels = batch
            print(images.type(), labels.type())
            with torch.no_grad():
                # images = torch.from_numpy(images).type(torch.FloatTensor).device()
                # labels = torch.from_numpy(labels).long().device()
                images = images.to(device)
                labels = labels.long().to(device)
            optimizer.zero_grad()
            torchV_lgt_171 = True
            if not torchV_lgt_171:
                outputs1, outputs2 = model_train(images, "train")
                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                _loss = _triplet_loss + _CE_loss

                _loss.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs1, outputs2 = model_train(images, "train")

                    _triplet_loss = loss(outputs1, Batch_size)
                    _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                    _loss = _triplet_loss + _CE_loss
                # ----------------------#
                #   反向传播
                # ----------------------#
                scaler.scale(_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            with torch.no_grad():
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))
            total_triple_loss += _triplet_loss.item()
            total_CE_loss += _CE_loss.item()
            total_accuracy += accuracy.item()

        save_model(model_train, "./model_data/", curr_epoch + e, total_CE_loss, total_accuracy)
