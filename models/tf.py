import torch
import torch.nn as nn

class TemporalFilter(nn.Module):
    def __init__(self, channels=64,
                 dw_kernel_size1=9, dw_kernel_size2=9,
                 sep_dw_kernel_size1=9, sep_pw_out_channels1=64, 
                 sep_dw_kernel_size2=9, sep_pw_out_channels2=64  
                ):
        super().__init__()
        self.channels = channels

        # --- 1. First Depthwise Convolution ---
        self.dw_conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=dw_kernel_size1,
            groups=channels,
            padding=0,       
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()

        # --- 2. Second Depthwise Convolution ---
        self.dw_conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=dw_kernel_size2,
            groups=channels,
            padding=0,      
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()

        current_channels_after_dw = channels

        # --- 3. First Separable Convolution (Depthwise + Pointwise) ---
        # 3a. Depthwise part
        self.sep_dw1 = nn.Conv1d(
            in_channels=current_channels_after_dw,
            out_channels=current_channels_after_dw,
            kernel_size=sep_dw_kernel_size1,
            groups=current_channels_after_dw,
            padding=0,    
            bias=False
        )
        self.bn_sep_dw1 = nn.BatchNorm1d(current_channels_after_dw)
        self.relu_sep_dw1 = nn.ReLU()

        # 3b. Pointwise part (1x1 convolution) 
        self.sep_pw1 = nn.Conv1d(
            in_channels=current_channels_after_dw,
            out_channels=sep_pw_out_channels1,
            kernel_size=1,
            bias=False
        )
        self.bn_sep_pw1 = nn.BatchNorm1d(sep_pw_out_channels1)
        self.relu_sep_pw1 = nn.ReLU()

        current_channels_after_sep1 = sep_pw_out_channels1

        # --- 4. Second Separable Convolution (Depthwise + Pointwise) ---
        # 4a. Depthwise part
        self.sep_dw2 = nn.Conv1d(
            in_channels=current_channels_after_sep1,
            out_channels=current_channels_after_sep1,
            kernel_size=sep_dw_kernel_size2,
            groups=current_channels_after_sep1,
            padding=0,     
            bias=False
        )
        self.bn_sep_dw2 = nn.BatchNorm1d(current_channels_after_sep1)
        self.relu_sep_dw2 = nn.ReLU()

        # 4b. Pointwise part
        self.sep_pw2 = nn.Conv1d(
            in_channels=current_channels_after_sep1,
            out_channels=sep_pw_out_channels2,
            kernel_size=1,
            bias=False
        )
        self.bn_sep_pw2 = nn.BatchNorm1d(sep_pw_out_channels2)
        self.relu_sep_pw2 = nn.ReLU()

    def forward(self, x):
        # Input x: (batch_size, channels, timepoints)

        # 1. First Depthwise Conv
        x = self.dw_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 2. Second Depthwise Conv
        x = self.dw_conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 3. First Separable Conv
        x_sep = self.sep_dw1(x)
        x_sep = self.bn_sep_dw1(x_sep)
        x_sep = self.relu_sep_dw1(x_sep)
        x_sep = self.sep_pw1(x_sep)
        x_sep = self.bn_sep_pw1(x_sep)
        x = self.relu_sep_pw1(x_sep)

        # 4. Second Separable Conv
        x_sep = self.sep_dw2(x)
        x_sep = self.bn_sep_dw2(x_sep)
        x_sep = self.relu_sep_dw2(x_sep)
        x_sep = self.sep_pw2(x_sep)
        x_sep = self.bn_sep_pw2(x_sep)
        x = self.relu_sep_pw2(x_sep)

        return x

# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 4
    channels = 64
    timepoints_in = 160

    eeg_signal = torch.randn(batch_size, channels, timepoints_in)
    print(f"Input Shape: {eeg_signal.shape}")

    temporal_filter_modified = TemporalFilter(
        channels=channels,
        dw_kernel_size1=9, dw_kernel_size2=9,
        sep_dw_kernel_size1=9, sep_pw_out_channels1=channels,
        sep_dw_kernel_size2=9, sep_pw_out_channels2=channels
    )

    output_modified = temporal_filter_modified(eeg_signal)
    print(f"Output Shape: {output_modified.shape}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {count_parameters(temporal_filter_modified)}")