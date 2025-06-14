import torch
import torch.nn as nn

class EEGSignalDecoder(nn.Module):
    def __init__(self,
                 input_channels_per_signal=64,
                 num_signals_to_concat=2,
                 output_channels=64,
                 hidden_dim=128,
                 timepoints=160,
                 intermediate_channels_list=None, 
                 kernel_sizes_list=None,     
                 final_activation=None):  
        super().__init__()

        self.input_channels_per_signal = input_channels_per_signal
        self.num_signals_to_concat = num_signals_to_concat
        self.output_channels = output_channels

        current_channels = input_channels_per_signal * num_signals_to_concat

        if intermediate_channels_list is None:
            intermediate_channels_list = [(current_channels + output_channels) // 2]
        if kernel_sizes_list is None:
            kernel_sizes_list = [5] * (len(intermediate_channels_list) + 1) 

        if len(kernel_sizes_list) != len(intermediate_channels_list) + 1:
            raise ValueError("kernel_sizes_list must have one more element than intermediate_channels_list "
                             "(for the final convolution layer mapping to output_channels).")

        layers = []

        for i, (out_ch, k_size) in enumerate(zip(intermediate_channels_list, kernel_sizes_list[:-1])):
            layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    kernel_size=k_size,
                    padding='same', 
                    bias=False 
                )
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU())
            current_channels = out_ch

        layers.append(
            nn.Conv1d(
                in_channels=current_channels,
                out_channels=output_channels,
                kernel_size=kernel_sizes_list[-1],
                padding='same',
                bias=True if final_activation is None else False
            )
        )

        if final_activation is not None:
            if not isinstance(final_activation, nn.Identity): 
                if layers[-1].bias is not None and layers[-1].bias.requires_grad:
                    layers[-1].bias = None 
                    print("Warning: Bias of final conv layer was set to True, but final_activation is provided. "
                          "Setting bias to False and adding BatchNorm if appropriate.")

                # Add BN before non-identity activation
                if not (isinstance(final_activation, nn.Tanh) or isinstance(final_activation, nn.Sigmoid)):
                    layers.append(nn.BatchNorm1d(output_channels))

            layers.append(final_activation)
        else:
            if layers[-1].bias is None:
                 layers[-1].bias = nn.Parameter(torch.zeros(output_channels))


        self.decoder_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, timepoints)

    def forward(self, signal_list):
        """
        Args:
            signal_list (list of Tensors): A list of tensors, e.g., [signal1, signal2].
                                           Each tensor is BxCxT (e.g., Bx64x160).
        Returns:
            Tensor: Reconstructed signal of shape Bxoutput_channelsxT (e.g., Bx64x160).
        """
        if not isinstance(signal_list, list) or len(signal_list) != self.num_signals_to_concat:
            raise ValueError(f"Input must be a list of {self.num_signals_to_concat} tensors.")

        for i, sig in enumerate(signal_list):
            if sig.shape[1] != self.input_channels_per_signal:
                raise ValueError(f"Signal {i} has {sig.shape[1]} channels, "
                                 f"expected {self.input_channels_per_signal}.")
            if i > 0 and sig.shape != signal_list[0].shape:
                raise ValueError("All input signals must have the same shape.")

        concatenated_signal = torch.cat(signal_list, dim=1)

        reconstructed_signal = self.decoder_layers(concatenated_signal)
        reconstructed_signal = self.linear(reconstructed_signal)
        return reconstructed_signal

# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 8
    channels_per_signal = 64
    timepoints = 128
    output_channels = 64

    eeg_signal1 = torch.randn(batch_size, channels_per_signal, timepoints)
    eeg_signal2 = torch.randn(batch_size, channels_per_signal, timepoints)
    print(f"Shape of eeg_signal1: {eeg_signal1.shape}")
    print(f"Shape of eeg_signal2: {eeg_signal2.shape}")

    decoder = EEGSignalDecoder(
        input_channels_per_signal=channels_per_signal,
        num_signals_to_concat=2,
        output_channels=output_channels,
        intermediate_channels_list=[], 
        kernel_sizes_list=[1],   
        final_activation=None
    )
    print("\nDecoder (Single 1x1 Conv):")
    print(decoder)
    reconstructed_signal = decoder([eeg_signal1, eeg_signal2])
    print(f"Shape of reconstructed_signal: {reconstructed_signal.shape}")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nParameters in decoder: {count_parameters(decoder)}")