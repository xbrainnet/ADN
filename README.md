# Adversarial Disentanglement Network (ADN)
Official codes for paper "Disentangled Representation Learning for Robust Brainprint Recognition"

## Contributions
![Network Architecture]([/\figs\model.png](https://github.com/xbrainnet/ADN/blob/main/figs/model.png?raw=true))

- Propose a generalized feature disentanglement framework for the adaptive extraction of intrinsic identity-related discriminative representation against bias representation
- Develop a novel spatial-temporal attention module to capture the spatial-temporal patterns within EEG signals, and further present a correlation-driven loss to preliminarily disentangle the entangled identity-bias factors. Then, the disentanglement is further improved by complementary constraint and adversarial training.
- Supervised contrastive learning strategy is extended to the component level to induce independence between the latent representations.

## Recommended System Configurations

- Python >=3.8
- CUDA 11
- Nvidia GPU with 12 GB ram at least

## Datasets

- [PhysioNet](https://www.physionet.org/content/eegmmidb/1.0.0/)
- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html)
- [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)
- [TUH-EEG](https://isip.piconepress.com/projects/tuh_eeg/)

## Example Usage of the ADN

```python
import torch
from models.tf import TemporalFilter
from models.decoder import EEGSignalDecoder

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
```

## EEG Pre-processing

Pre-process the EEG signals as mentioned in the section IV. Experiments-A. Dataset and Pre-processing.

## Training

Getting the identification results and model weights by runing main.py 

## Citation

If you use this code, please cite the corresponding paper: (unavailable now)
