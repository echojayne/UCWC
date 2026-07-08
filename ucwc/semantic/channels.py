from dataclasses import dataclass

import numpy as np
import torch, math

class Channels():
    """
    This class simulates various wireless communication channels, including
    Additive White Gaussian Noise (AWGN), Rayleigh fading, and Rician fading channels.

    Attributes:
    device : torch.device
        Specifies the device (CPU or GPU) on which computations will be performed.

    Methods:
    __init__(device):
        Initializes the class with a specified device (CPU or GPU).

    calculate_signal_power(Tx_sig):
        Computes the power of the transmitted signal (Tx_sig) by taking the mean of the square of the signal.

    AWGN(Tx_sig, snr, H=None):
        Adds Additive White Gaussian Noise (AWGN) to the transmitted signal `Tx_sig`.
        Optionally, it can consider channel fading effects (`H`) if provided.
        The Signal-to-Noise Ratio (SNR) is specified in dB. The function returns the received signal after adding noise.

    Rayleigh(Tx_sig, snr):
        Simulates a Rayleigh fading channel by generating random channel coefficients
        and applying the fading effect to the transmitted signal `Tx_sig`. Then, AWGN is added,
        and the channel is estimated to recover the received signal `Rx_sig`.

    Rician(Tx_sig, snr, K=1):
        Simulates a Rician fading channel, where the fading coefficients are generated
        based on the specified `K` factor. A `K` factor of 1 means the channel is more
        similar to Rayleigh fading. The function adds AWGN and performs channel estimation to
        recover the received signal `Rx_sig`.
    """
    def __init__(self, device):
        self.device = device

    def calculate_signal_power(self, Tx_sig):
        return torch.mean(Tx_sig ** 2)

    def AWGN(self, Tx_sig, snr, H=None):
        if H is not None:
            shape = Tx_sig.shape
            Tx_sig_without_channel = torch.matmul(Tx_sig, torch.inverse(H)).view(shape)
            signal_power = self.calculate_signal_power(Tx_sig_without_channel)
        else:
            signal_power = self.calculate_signal_power(Tx_sig)
        n_var = signal_power / 10**(snr / 10)
        Rx_sig = Tx_sig + torch.normal(0, math.sqrt(n_var), size=Tx_sig.shape).to(self.device)
        return Rx_sig

    def Rayleigh(self, Tx_sig, snr):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)

        Rx_sig = self.AWGN(Tx_sig, snr, H)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape) # Channel estimation

        return Rx_sig

    def Rician(self, Tx_sig, snr, K=1):
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(self.device)
        H_imag = torch.normal(mean, std, size=[1]).to(self.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)

        Rx_sig = self.AWGN(Tx_sig, snr, H)
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape) # Channel estimation

        return Rx_sig


@dataclass(frozen=True, slots=True)
class ChannelResult:
    received_symbols: np.ndarray
    noise_variance: float
    channel_gain: np.ndarray


def simulate_channel(
    symbols: np.ndarray,
    *,
    snr_db: float,
    channel_model: str = "awgn",
    seed: int = 0,
) -> ChannelResult:
    """Adapt the project QAM complex-symbol path to the updated torch channel."""

    tx = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if len(tx) == 0:
        raise ValueError("symbols must not be empty.")

    device = torch.device("cpu")
    tx_real = torch.from_numpy(np.stack([tx.real, tx.imag], axis=-1)).float().to(device)
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        channel = Channels(device)
        model = channel_model.lower().strip()
        if model == "awgn":
            rx_real = channel.AWGN(tx_real, float(snr_db))
        elif model == "rayleigh":
            rx_real = channel.Rayleigh(tx_real, float(snr_db))
        elif model == "rician":
            rx_real = channel.Rician(tx_real, float(snr_db))
        else:
            raise ValueError("channel_model must be 'awgn', 'rayleigh', or 'rician'.")
    finally:
        torch.random.set_rng_state(state)

    rx_np = rx_real.detach().cpu().numpy().reshape(-1, 2)
    received = rx_np[:, 0].astype(np.float64) + 1j * rx_np[:, 1].astype(np.float64)
    noise = received - tx
    noise_variance = max(float(np.mean(np.abs(noise) ** 2)), np.finfo(np.float64).tiny)
    return ChannelResult(
        received_symbols=received,
        noise_variance=noise_variance,
        channel_gain=np.ones_like(received, dtype=np.complex128),
    )
