import torch
import torch.nn as nn
from models.tf import TemporalFilter
from models.decoder import EEGSignalDecoder

class STA(nn.Module):
    def __init__(self, channels, hidden_dim, embedding_dim=128):
        super(STA, self).__init__()
        self.spatial_projection = nn.Linear(hidden_dim, embedding_dim)
        self.temporal_projection = nn.Linear(channels, embedding_dim)

    def forward(self, x): 
        res = x
        Queries = self.spatial_projection(x)
        Keys = self.temporal_projection(x.transpose(1, 2))
        S = torch.matmul(Queries, Keys.transpose(1, 2))
        S = S / (Keys.shape[-1] ** 0.5)
        S = torch.nn.functional.softmax(S, dim=-1)
        x = S * x
        x = x + res
        return x, S

class IdentityEnc(nn.Module):
    def __init__(self, channels,
                 dw_kernel_size1, dw_kernel_size2, 
                 sep_dw_kernel_size1, sep_pw_out_channels1, 
                 sep_dw_kernel_size2, sep_pw_out_channels2,
                 hidden_dim=128):
        super(IdentityEnc, self).__init__()

        self.filter = TemporalFilter(channels, 
                                     dw_kernel_size1, dw_kernel_size2, 
                                     sep_dw_kernel_size1, sep_pw_out_channels1, 
                                     sep_dw_kernel_size2, sep_pw_out_channels2)
        self.attention = STA(channels, hidden_dim)

    def forward(self, x):
        x_filtered = self.filter(x)
        z_identity, S_identity = self.attention(x_filtered)

        return x, z_identity, S_identity
    
class BiasEnc(nn.Module):
    def __init__(self, channels,
                 dw_kernel_size1, dw_kernel_size2, 
                 sep_dw_kernel_size1, sep_pw_out_channels1, 
                 sep_dw_kernel_size2, sep_pw_out_channels2,
                 hidden_dim=128):
        super(BiasEnc, self).__init__()

        self.filter = TemporalFilter(channels, 
                                     dw_kernel_size1, dw_kernel_size2, 
                                     sep_dw_kernel_size1, sep_pw_out_channels1, 
                                     sep_dw_kernel_size2, sep_pw_out_channels2)
        self.attention = STA(channels, hidden_dim)

    def forward(self, x):
        x_filtered = self.filter(x)
        z_bias, S_bias = self.attention(x_filtered)

        return z_bias, S_bias

class Decoder(nn.Module):
    def __init__(self,
                 channels=64,
                 output_channels=64,
                 kernel_size=1):    
        super().__init__()

        self.decoder_layers = nn.Conv1d(
                in_channels=channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding='same'
            )

    def forward(self, signal_list):
        print(signal_list[0].shape, signal_list[1].shape)
        concatenated_signal = torch.cat(signal_list, dim=1)
        print(f"Concatenated signal shape: {concatenated_signal.shape}")
        reconstructed_signal = self.decoder_layers(concatenated_signal)

        return reconstructed_signal

class IdentityCls(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(IdentityCls, self).__init__()
        
        self.tsconv = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (63, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(880, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.tsconv(x.unsqueeze(1))
        x = self.classifier(torch.flatten(x, 1))
        return x
    
class BiasCls(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(BiasCls, self).__init__()

        self.tsconv = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (63, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(880, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.tsconv(x.unsqueeze(1))
        x = self.classifier(torch.flatten(x, 1))
        return x

# --- Example Usage ---

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.randn(40, 64, 160).to(device)
    x2 = torch.randn(40, 64, 160).to(device)
    channels = 64
    timepoints = 160
    hidden_dim = 128
    num_classes = 105

    identity_enc = IdentityEnc(channels=channels,
                             dw_kernel_size1=9, dw_kernel_size2=9,
                             sep_dw_kernel_size1=9, sep_pw_out_channels1=channels,
                             sep_dw_kernel_size2=9, sep_pw_out_channels2=channels,
                             hidden_dim=hidden_dim).to(device)
    bias_enc = BiasEnc(channels=channels,
                        dw_kernel_size1=9, dw_kernel_size2=9,
                        sep_dw_kernel_size1=9, sep_pw_out_channels1=channels,
                        sep_dw_kernel_size2=9, sep_pw_out_channels2=channels,
                        hidden_dim=hidden_dim).to(device)

    identity_cls = IdentityCls(hidden_dim, num_classes).to(device)
    bias_cls = BiasCls(hidden_dim, num_classes).to(device)

    decoder = EEGSignalDecoder(input_channels_per_signal=channels, num_signals_to_concat=2,
                               output_channels=channels, hidden_dim=hidden_dim,
                               timepoints=timepoints, intermediate_channels_list=[],
                               kernel_sizes_list=[1], final_activation=None).to(device)

    model = [identity_enc, bias_enc, identity_cls, bias_cls, decoder]

    x, z_identity, S_identity = model[0](x1)
    z_bias, S_bias = model[1](x1)
    logits_i = model[2](z_identity)
    logits_b = model[3](z_bias)
    x_hat = model[4]([z_identity, z_bias])
    print(x.shape, z_identity.shape, 
          S_identity.shape, z_bias.shape, 
          S_bias.shape, logits_i.shape, 
          logits_b.shape, x_hat.shape)
