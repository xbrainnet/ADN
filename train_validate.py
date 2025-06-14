import time
import torch
import torch.nn as nn
from utils import Metrics

def train(model_q, model_k, optimizer, scheduler, criterion, 
          train_loader, valid_loader, device, num_epoch, fold, alpha, beta, init=False):
    def init_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight.data)

    if init:
        model_q.apply(init_kaiming)
        model_k.apply(init_kaiming)
    
    best_acc, f1, = 0.0, 0.0
    train_acces, train_losses_Ei, train_losses_Eb, train_losses_Cb, valid_acces, valid_losses = [], [], [], [], [], []

    for epoch in range(num_epoch):
        start = time.time()
        train_epoch_loss_Ei = 0
        train_epoch_loss_Eb = 0
        train_epoch_loss_Cb = 0
        valid_epoch_loss = 0

        model_q[0].train(), model_k[0].train()
        model_q[1].train(), model_k[1].train()
        model_q[2].train(), model_k[2].train()
        model_q[3].train(), model_k[3].train()
        model_q[4].train(), model_k[4].train()

        # train
        predict_train, pre_train, y_true_train = None, list(), list()
        for i, (x1, x2, label) in enumerate(train_loader):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            
            x, z_identity, S_identity = model_q[0](x1)
            z_bias, S_bias = model_q[1](x1)
            logits_i = model_q[2](z_identity)
            logits_b = model_q[3](z_bias)
            x_hat = model_q[4]([z_identity, z_bias])

            # Augmentation Branch
            _, z_identity_aug, _ = model_k[0](x2)
            z_bias_aug, _ = model_k[1](x2)

            # Loss Calculation
            loss_identityCls = criterion[0](logits_i, label.long())
            loss_biasCls = criterion[0](logits_b, label.long())
            loss_cc = criterion[2](z_identity.reshape(1, -1), z_bias.reshape(1, -1)) + criterion[2](S_identity.reshape(1, -1), S_bias.reshape(1, -1))
            loss_recon = criterion[1](x.reshape(1, -1), x_hat.reshape(1, -1))
            loss_csupcon = criterion[3](z_identity, z_identity_aug, z_bias, z_bias_aug, label.long())

            loss_Ei = loss_identityCls + alpha * (loss_recon + loss_cc) + beta * loss_csupcon
            loss_Eb = -loss_biasCls + alpha * (loss_recon + loss_cc) + beta * loss_csupcon
            loss_Cb = criterion[0](logits_b, label.long())
            
            # train Ei, Ci and Decoder
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            optimizer[2].zero_grad()
            
            loss_Ei.requires_grad_(True)
            loss_Eb.requires_grad_(True)
            loss_Cb.requires_grad_(True)
            loss_Ei.backward(retain_graph=True)
            loss_Eb.backward(retain_graph=True)
            loss_Cb.backward(retain_graph=True)

            optimizer[0].step()
            optimizer[1].step()
            optimizer[2].step()
            
            train_epoch_loss_Ei += loss_Ei.item()
            train_epoch_loss_Eb += loss_Eb.item()
            train_epoch_loss_Cb += loss_Cb.item()

            predict_train = logits_i.max(1)[1]
            
            pre_train.extend(predict_train.cpu().numpy())
            y_true_train.extend(label.cpu().numpy())

        scheduler[0].step()
        scheduler[1].step()
        scheduler[2].step()
        train_acc, _, = Metrics(y_true_train, pre_train)
        
        if epoch % 10 == 0:
            torch.save(model_q[0].state_dict(), 'checkpoint/Fold_' + str(fold) + '_epoch' + str(epoch) + '_Ei.pth')
            torch.save(model_q[1].state_dict(), 'checkpoint/Fold_' + str(fold) + '_epoch' + str(epoch) + '_Eb.pth')
            torch.save(model_q[2].state_dict(), 'checkpoint/Fold_' + str(fold) + '_epoch' + str(epoch) + '_Ci.pth')
        
        # validation
        predict, pre, y_true = None, list(), list()
        model_q[0].eval()  
        model_q[1].eval()
        model_q[2].eval()
        model_q[3].eval()
        model_q[4].eval()     
        with torch.no_grad():
            for i, (x1, x2, label) in enumerate(valid_loader):
                x1, label = x1.to(device), label.to(device)

                x, z_identity, S_identity = model_q[0](x1)
                z_bias, S_bias = model_q[1](x1)
                logits_i = model_q[2](z_identity)
                logits_b = model_q[3](z_bias)
                x_hat = model_q[4]([z_identity, z_bias])

                loss = criterion[0](logits_i, label.long()) + criterion[1](x.reshape(1, -1), x_hat.reshape(1, -1)) + criterion[2](z_identity.reshape(1, -1), z_bias.reshape(1, -1))
                valid_epoch_loss += loss.item()

                predict = logits_i.max(1)[1]
                
                pre.extend(predict.cpu().numpy())
                y_true.extend(label.cpu().numpy())

            valid_acc, valid_f1 = Metrics(y_true, pre)
            if valid_acc > best_acc:
                best_acc = valid_acc
                f1 = valid_f1
                torch.save(model_q[0].state_dict(), 'checkpoint/Fold_' + str(fold) + '_best_Ei.pth')
                torch.save(model_q[1].state_dict(), 'checkpoint/Fold_' + str(fold) + '_best_Eb.pth')
                torch.save(model_q[2].state_dict(), 'checkpoint/Fold_' + str(fold) + '_best_Ci.pth')

            train_epoch_loss_Ei = train_epoch_loss_Ei / len(train_loader)
            train_epoch_loss_Eb = train_epoch_loss_Eb / len(train_loader)
            train_epoch_loss_Cb = train_epoch_loss_Cb / len(train_loader)
            valid_epoch_loss = valid_epoch_loss / len(valid_loader)

            end = time.time() - start

        train_acces.append(train_acc)
        train_losses_Ei.append(train_epoch_loss_Ei)
        train_losses_Eb.append(train_epoch_loss_Eb)
        train_losses_Cb.append(train_epoch_loss_Cb)
        valid_acces.append(valid_acc)
        valid_losses.append(valid_epoch_loss)
        print("< Fold{} {:.0f}% {}/{} {:.3f}s >".format(fold, (epoch + 1) / num_epoch * 100, epoch + 1, num_epoch, end), end="")
        print('train_loss_Ei =', '{:.5f}'.format(train_epoch_loss_Ei), end=" ")
        print('train_loss_Eb =', '{:.5f}'.format(train_epoch_loss_Eb), end=" ")
        print('train_loss_Cb =', '{:.5f}'.format(train_epoch_loss_Cb), end=" ")
        print('train_acc =', '{:.5f}'.format(train_acc), end=" ")
        print('valid_loss =', '{:.5f}'.format(valid_epoch_loss), end=" ")
        print('valid_acc =', '{:.4f}'.format(valid_acc))
    
    return best_acc, f1, train_acces, train_losses_Ei, train_losses_Eb, train_losses_Cb, valid_acces, valid_losses