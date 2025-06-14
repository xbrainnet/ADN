import argparse

class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        ################******************** data settings ***************###########################

        self.parser.add_argument('--path_data', type=str, default='F:\Papers\EEG-Authentication\Paper-main\data_PhysioNet_1s.npy', help='path of data')
        self.parser.add_argument('--path_label', type=str, default='F:\Papers\EEG-Authentication\Paper-main\label_PhysioNet_1s.npy', help='path of label')
        self.parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training [default: 128]')
        self.parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train [default: 500]')
        self.parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='initial learning rate of adam [default: 200]')

        ################******************** loader settings ***************###########################
        
        self.parser.add_argument('--contra', type=int, default=1, help='whether to use contrastive learning')
        self.parser.add_argument('--n_channels', type=int, default=64, help='number of EEG channels')
        self.parser.add_argument('--bound', type=float, default=0.00025, help='bound of noise')
        self.parser.add_argument('--bandpass1', type=list, default=[1, 30], help='first bandpass filter for augmentation')
        self.parser.add_argument('--bandpass2', type=list, default=[10, 40], help='second bandpass filter for augmentation')
        self.parser.add_argument('--signal_freq', type=int, default=160, help='frequency of EEG signal')

        ################******************** model settings ***************###########################
        self.parser.add_argument('--channels', type=int, default=64, help='input dim of cnn encoder')
        self.parser.add_argument('--timepoints', type=int, default=160, help='timestamp dim of data after stft')
        self.parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim of classifier')
        self.parser.add_argument('--num_classes', type=int, default=105, help='number of label classes')

        ################********************** Loss settings **************############################
        self.parser.add_argument('--alpha', type=float, default=0.4, help='parameter for CC and Recon loss')
        self.parser.add_argument('--beta', type=float, default=0.6, help='parameter for C-SupCon loss')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

