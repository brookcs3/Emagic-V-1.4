"""Multi-resolution STFT loss and combined training loss for BSRoformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Compute L1 loss on magnitude spectrograms at multiple STFT resolutions.

    Each resolution captures a different time-frequency trade-off:
      - Large windows  → fine frequency resolution (harmonics)
      - Small windows  → fine time resolution (transients)
    """

    def __init__(
        self,
        resolutions: list[tuple[int, int, int]] = (
            (4096, 1024, 4096),
            (2048, 512, 2048),
            (1024, 256, 1024),
            (512, 128, 512),
            (256, 64, 256),
        ),
    ):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, T] or [B, T] predicted waveform
            target: [B, C, T] or [B, T] ground truth waveform
        Returns:
            Scalar loss averaged over all resolutions.
        """
        if pred.dim() == 3:
            B, C, T = pred.shape
            pred = pred.reshape(B * C, T)
            target = target.reshape(B * C, T)

        loss = torch.tensor(0.0, device=pred.device, dtype=torch.float32)

        for n_fft, hop_length, win_length in self.resolutions:
            window = torch.hann_window(win_length, device=pred.device)

            pred_stft = torch.stft(
                pred, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )
            target_stft = torch.stft(
                target, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                window=window, return_complex=True,
            )

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            loss = loss + F.l1_loss(pred_mag, target_mag)

        return loss / len(self.resolutions)


def combined_loss(
    pred_stems: torch.Tensor,
    target_stems: torch.Tensor,
    mrstft: MultiResolutionSTFTLoss,
    l1_weight: float = 1.0,
    mrstft_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined L1 time-domain + multi-resolution STFT loss.

    Args:
        pred_stems:   [B, num_stems, 2, T]
        target_stems: [B, num_stems, 2, T]
        mrstft:       MultiResolutionSTFTLoss instance
        l1_weight:    weight for time-domain L1
        mrstft_weight: weight for multi-resolution STFT loss

    Returns:
        Dict with 'total', 'l1', and 'mrstft' loss tensors.
    """
    l1 = F.l1_loss(pred_stems, target_stems)

    B, S, C, T = pred_stems.shape
    pred_flat = pred_stems.reshape(B * S, C, T)
    target_flat = target_stems.reshape(B * S, C, T)
    stft_loss = mrstft(pred_flat, target_flat)

    total = l1_weight * l1 + mrstft_weight * stft_loss

    return {"total": total, "l1": l1, "mrstft": stft_loss}
