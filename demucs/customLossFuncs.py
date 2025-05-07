## Addition to the Demucs. Made by Mads Lang Laursen. 

import torch.nn.functional as F
import torch.nn as nn
import torch as th
from itertools import permutations

from .parser import get_parser

parser = get_parser()
args = parser.parse_args()



class SilenceWeightedMSELoss(nn.Module):
    def __init__(self, silence_threshold=0.001, silence_weight=0.1):
        super(SilenceWeightedMSELoss, self).__init__()
        self.silence_threshold = silence_threshold
        self.silence_weight = silence_weight

    def forward(self, input, target):
        input = th.clamp(input, min=-1e6, max=1e6)
        target = th.clamp(target, min=-1e6, max=1e6)

        silence_mask = (target.abs() < self.silence_threshold).float()
        weight_silence = silence_mask * self.silence_weight + (1 - silence_mask)
        loss = (weight_silence * (input - target) ** 2).mean()
        return loss


class CCMSE(th.nn.Module):
    """complex STFT loss module."""

    def __init__(
        self,
        fft_size=1024, 
        shift_size=120, 
        win_length=600,
        window="hann_window",
        alpha=args.alpha, 
        c = args.comp_factor, 
        **kwargs,
    ):
        """Initialize STFT loss module."""
        super(CCMSE, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(th, window)(win_length))
        self.alpha = alpha
        self.c = c

    def orig_ccmse(self, y_speaker_channel, y_target_speaker_channel, epsilon=1e-6):
        y_stft = self.cstft(
            y_speaker_channel, 
            self.fft_size, 
            self.shift_size, 
            self.win_length, 
            self.window.to(y_speaker_channel.device)
        )

        y_target_stft = self.cstft(
            y_target_speaker_channel,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window.to(y_target_speaker_channel.device),
        )

        mag_y = th.pow(th.abs(y_stft) + epsilon, self.c)
        mag_y_target = th.pow(th.abs(y_target_stft) + epsilon, self.c)
        unit_y = th.exp(1j * th.angle(y_stft))
        unit_y_target = th.exp(1j * th.angle(y_target_stft))

        mag_y = th.clamp(mag_y, min=epsilon)
        mag_y_target = th.clamp(mag_y_target, min=epsilon)

        mag_loss = th.abs(mag_y - mag_y_target) ** 2
        c_loss = th.abs(mag_y * unit_y - mag_y_target * unit_y_target) ** 2
        loss = th.mean(self.alpha * c_loss + (1 - self.alpha) * mag_loss)
        return loss
    


    def logcomp(self, y_true, y_pred):
        """Non-stft version of the logcomp loss function."""
        y_true_mag = th.log1p(th.abs(y_true))
        y_pred_mag = th.log1p(th.abs(y_pred))
        
        y_true_phase = th.angle(y_true)
        y_pred_phase = th.angle(y_pred)
        
        y_true_combined = y_true_mag * th.exp(1j * y_true_phase)
        y_pred_combined = y_pred_mag * th.exp(1j * y_pred_phase)
        
        loss = th.mean(th.abs(y_true_combined - y_pred_combined) ** 2)

        loss = loss.contiguous()
        return loss
    
    def logcomp_frequency(self, y_speaker_channel, y_target_speaker_channel):
        y_stft = self.cstft(
            y_speaker_channel, 
            self.fft_size, 
            self.shift_size, 
            self.win_length, 
            self.window.to(y_speaker_channel.device)
        )

        y_target_stft = self.cstft(
            y_target_speaker_channel,
            self.fft_size,
            self.shift_size,
            self.win_length,
            self.window.to(y_target_speaker_channel.device),
        )

        y_true_mag = th.atan(5*th.abs(y_target_stft))
        y_pred_mag = th.atan(5*th.abs(y_stft))
        
        y_true_phase = th.angle(y_target_stft)
        y_pred_phase = th.angle(y_stft)

        y_true_combined = y_true_mag * th.exp(1j * y_true_phase)
        y_pred_combined = y_pred_mag * th.exp(1j * y_pred_phase)

        loss = th.mean(th.abs(y_true_combined - y_pred_combined) ** 2)
        loss = loss.contiguous()
        return loss


    def forward(self, y, y_target):
        """Calculate forward propagation.

        Args:
            y (Tensor): Predicted signal (B, T).
            y_target (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Average loss value across all speakers.
        """

        if y.dim() == 3:   
            num_speakers, num_channels, num_samples = y.shape
        if y.dim() == 4:
            batch_size, num_speakers, num_channels, num_samples = y.shape

        total_loss = 0.0
        


        for speaker in range(num_speakers):
            for channel in range(num_channels):
                if y.dim() == 3:
                    y_speaker_channel = y[speaker, channel, :]
                    y_target_speaker_channel = y_target[speaker, channel, :]
                if y.dim() == 4:
                    y_speaker_channel = y[:, speaker, channel, :]
                    y_target_speaker_channel = y_target[:, speaker, channel, :]
                
                
                
                if args.logcompFreq:
                    total_loss += self.logcomp_frequency(y_speaker_channel, y_target_speaker_channel)
                elif args.logcomp:
                    total_loss += self.logcomp(y_speaker_channel, y_target_speaker_channel)
                else:
                    total_loss += self.orig_ccmse(y_speaker_channel, y_target_speaker_channel)
            
            average_loss = total_loss / (num_speakers * num_channels)
        return average_loss

    def cstft(self, x, fft_size, hop_size, win_length, window):
        """Perform STFT

        Args:
            x (Tensor): Input signal tensor (B, T).
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.

        Returns:
            Tensor: real spectrogram (B, #frames, fft_size // 2 + 1).
            Tensor: imag spectrogram (B, #frames, fft_size // 2 + 1).

        """
        x_stft = th.stft(
            x,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_length,
            window=window,
            return_complex=True)
        return x_stft

class SI_SDR(th.nn.Module):
    def __init__(self, epsilon=1e-7):
        """SI-SDR loss module."""
        super(SI_SDR, self).__init__()
        self.epsilon = epsilon

    def si_sdr_loss(self, y, y_target):
        """Compute SI-SDR loss for a single pair of signals."""
        y = y - y.mean(dim=-1, keepdim=True)
        y_target = y_target - y_target.mean(dim=-1, keepdim=True)
        
        s_target_num = th.sum(y * y_target, dim=-1, keepdim=True) * y_target
        s_target_den = th.sum(y_target ** 2, dim=-1, keepdim=True) + self.epsilon
        s_target = s_target_num / s_target_den
        
        e_noise = y - s_target
        
        s_target_energy = th.sum(s_target ** 2, dim=-1, keepdim=True)
        e_noise_energy = th.sum(e_noise ** 2, dim=-1, keepdim=True)
        
        s_target_energy = th.clamp(s_target_energy, min=self.epsilon)
        e_noise_energy = th.clamp(e_noise_energy, min=self.epsilon)
        
        si_sdr = 10 * th.log10(s_target_energy / e_noise_energy)
        
        return -si_sdr.mean()

    def forward(self, y, y_target):
        """Permutation-invariant SI-SDR loss.

        Args:
            y (Tensor): Predicted signal (B, S, T), where S is the number of speakers.
            y_target (Tensor): Groundtruth signal (B, S, T).

        Returns:
            Tensor: Minimum SI-SDR loss across all permutations.
        """
        batch_size, num_speakers, _ = y.shape
        perms = list(permutations(range(num_speakers)))
        min_loss = None

        for perm in perms:
            loss = 0
            for i, j in enumerate(perm):
                loss += self.si_sdr_loss(y[:, i, :], y_target[:, j, :])
            loss = loss / num_speakers

            if min_loss is None or loss < min_loss:
                min_loss = loss

        return min_loss


class PIT_SI_SDR(th.nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        """
        Compute permutation-invariant SI-SDR loss.

        Args:
            preds: Tensor of shape (B, N, C, T) or (B, N, T) - predicted signals
            targets: Tensor of shape (B, N, C, T) or (B, N, T) - ground truth signals
                     where B = batch size, N = number of sources, C = channels (optional), T = time steps

        Returns:
            Scalar loss (negative SI-SDR averaged over best permutation)
        """
        if preds.dim() == 3:  # If input is (B, N, T), add a channel dimension
            preds = preds.unsqueeze(2)  # (B, N, 1, T)
            targets = targets.unsqueeze(2)  # (B, N, 1, T)

        B, N, C, T = preds.shape
        assert preds.shape == targets.shape, "Shape mismatch between preds and targets"

        # Normalize to zero-mean along the time dimension
        preds = preds - preds.mean(dim=3, keepdim=True)
        targets = targets - targets.mean(dim=3, keepdim=True)

        # Calculate pairwise SI-SDR between each prediction-target pair
        pairwise_si_sdr = []

        for perm in permutations(range(N)):
            permuted_targets = targets[:, perm, :, :]  # (B, N, C, T)
            s_target = th.sum(preds * permuted_targets, dim=3, keepdim=True) * permuted_targets
            s_target = s_target / (th.sum(permuted_targets ** 2, dim=3, keepdim=True) + self.epsilon)

            e_noise = preds - s_target

            s_target_energy = th.sum(s_target ** 2, dim=3)
            e_noise_energy = th.sum(e_noise ** 2, dim=3)

            si_sdr = 10 * th.log10((s_target_energy + self.epsilon) / (e_noise_energy + self.epsilon))
            pairwise_si_sdr.append(si_sdr.mean(dim=(1, 2)))  # shape: (B,)

        # Stack SI-SDR for all permutations: shape (n_perms, B)
        si_sdr_all = th.stack(pairwise_si_sdr, dim=0)

        # Max SI-SDR (i.e., min loss) across permutations per example
        max_sisdr, _ = si_sdr_all.max(dim=0)

        return -max_sisdr.mean()  # Negative because we want to *maximize* SI-SDR
