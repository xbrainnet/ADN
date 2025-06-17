import torch
import itertools
import statistics
import torch.nn as nn
from params import Config
from utils import SetSeeds
from dataset import MIDataset
from train_validate import train
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from models.decoder import EEGSignalDecoder
from loss import CCLoss, ComponentWiseSupConLoss
from models.net import IdentityEnc, BiasEnc, IdentityCls, BiasCls

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

    seed = 3407
    print('seed is {}'.format(seed))
    print('training on:', device)
    SetSeeds(seed)

    args = Config().parse()

    dataset = MIDataset(path_data = args.path_data, path_label = args.path_label, 
                        contra=args.contra, bound=args.bound, 
                        bandpass1=args.bandpass1, bandpass2=args.bandpass2, 
                        signal_freq=args.signal_freq)

    K = 10
    KF = KFold(n_splits=K, shuffle=True, random_state=seed)

    predict_acc, predict_f1 = [], []

    fold = 1
    for train_idx, test_idx in KF.split(dataset):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_subsampler)

        identity_enc = IdentityEnc(channels=args.channels,
                             dw_kernel_size1=9, dw_kernel_size2=9,
                             sep_dw_kernel_size1=9, sep_pw_out_channels1=args.channels,
                             sep_dw_kernel_size2=9, sep_pw_out_channels2=args.channels,
                             hidden_dim=args.hidden_dim).to(device)
        bias_enc = BiasEnc(channels=args.channels,
                             dw_kernel_size1=9, dw_kernel_size2=9,
                             sep_dw_kernel_size1=9, sep_pw_out_channels1=args.channels,
                             sep_dw_kernel_size2=9, sep_pw_out_channels2=args.channels,
                             hidden_dim=args.hidden_dim).to(device)
        identity_cls = IdentityCls(args.hidden_dim, args.num_classes).to(device)
        bias_cls = BiasCls(args.hidden_dim, args.num_classes).to(device)
        decoder = EEGSignalDecoder(input_channels_per_signal=args.channels, num_signals_to_concat=2,
                               output_channels=args.channels, hidden_dim=args.hidden_dim,
                               timepoints=args.timepoints, intermediate_channels_list=[],
                               kernel_sizes_list=[1], final_activation=None).to(device)
        
        # Augmentation Branch
        identity_enc_aug = IdentityEnc(channels=args.channels,
                             dw_kernel_size1=9, dw_kernel_size2=9,
                             sep_dw_kernel_size1=9, sep_pw_out_channels1=args.channels,
                             sep_dw_kernel_size2=9, sep_pw_out_channels2=args.channels,
                             hidden_dim=args.hidden_dim).to(device)
        bias_enc_aug = BiasEnc(channels=args.channels,
                             dw_kernel_size1=9, dw_kernel_size2=9,
                             sep_dw_kernel_size1=9, sep_pw_out_channels1=args.channels,
                             sep_dw_kernel_size2=9, sep_pw_out_channels2=args.channels,
                             hidden_dim=args.hidden_dim).to(device)
        identity_cls_aug = IdentityCls(args.hidden_dim, args.num_classes).to(device)
        bias_cls_aug = BiasCls(args.hidden_dim, args.num_classes).to(device)
        decoder_aug = EEGSignalDecoder(input_channels_per_signal=args.channels, num_signals_to_concat=2,
                               output_channels=args.channels, hidden_dim=args.hidden_dim,
                               timepoints=args.timepoints, intermediate_channels_list=[],
                               kernel_sizes_list=[1], final_activation=None).to(device)
        
        model_q = [identity_enc, bias_enc, identity_cls, bias_cls, decoder]
        model_k = [identity_enc_aug, bias_enc_aug, identity_cls_aug, bias_cls_aug, decoder_aug]

        ce_loss = nn.CrossEntropyLoss()
        recon_loss = nn.MSELoss()
        cc_loss = CCLoss()
        csupcon_loss = ComponentWiseSupConLoss()
        criterion = [ce_loss, recon_loss, cc_loss, csupcon_loss]
        
        optimizer_IdentityEnc = torch.optim.Adam(itertools.chain(model_q[0].parameters(), model_q[2].parameters(), model_q[4].parameters()), lr=args.lr, weight_decay=1e-5)
        optimizer_BiasEnc = torch.optim.Adam(itertools.chain(model_q[1].parameters(), model_q[3].parameters(), decoder.parameters()), lr=args.lr, weight_decay=1e-5)
        optimizer_BiasCls = torch.optim.Adam(model_q[3].parameters(), lr=args.lr, weight_decay=1e-5)
        optimizer = [optimizer_IdentityEnc, optimizer_BiasEnc, optimizer_BiasCls]
        
        scheduler_IdentityEnc = torch.optim.lr_scheduler.ExponentialLR(optimizer_IdentityEnc, gamma=0.99)
        scheduler_BiasEnc = torch.optim.lr_scheduler.ExponentialLR(optimizer_BiasEnc, gamma=0.99)
        scheduler_BiasCls = torch.optim.lr_scheduler.ExponentialLR(optimizer_BiasCls, gamma=0.99)
        scheduler = [scheduler_IdentityEnc, scheduler_BiasEnc, scheduler_BiasCls]

        train_acces, train_losses_Ei, train_losses_Eb, train_losses_Cb, valid_acces, valid_losses = train(model_q, model_k, optimizer, scheduler, criterion, train_loader, val_loader, device, args.epochs, fold, args.alpha, args.beta)

        fold += 1
