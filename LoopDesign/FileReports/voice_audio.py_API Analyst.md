# API Analyst Report: voice\audio.py

## Dependencies
- `import os`
- `from functools import lru_cache`
- `from subprocess import CalledProcessError`
- `from subprocess import run`
- `from typing import Optional`
- `from typing import Union`
- `import numpy as np`
- `import torch`
- `import torch.nn.functional as F`

## Configuration Variables
- `SAMPLE_RATE` = `16000`
- `N_FFT` = `400`
- `HOP_LENGTH` = `160`
- `CHUNK_LENGTH` = `30`
- `N_SAMPLES` = `CHUNK_LENGTH * SAMPLE_RATE`
- `N_FRAMES` = `exact_div(N_SAMPLES, HOP_LENGTH)`
- `N_SAMPLES_PER_TOKEN` = `HOP_LENGTH * 2`
- `FRAMES_PER_SECOND` = `exact_div(SAMPLE_RATE, HOP_LENGTH)`
- `TOKENS_PER_SECOND` = `exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)`

## Functions & Endpoints

### `exact_div`
`def exact_div(x, y)`
### `load_audio`
`def load_audio(file: str, sr: int=SAMPLE_RATE)`
> Open an audio file and read as mono waveform, resampling as necessary

Parameters
----------
file: str
    The audio file to open

sr: int
    The sample rate to resample the audio if necessary

Returns
-------
A NumPy array containing the audio waveform, in float32 dtype.

### `pad_or_trim`
`def pad_or_trim(array, length: int=N_SAMPLES, *, axis: int=-1)`
> Pad or trim the audio array to N_SAMPLES, as expected by the encoder.

### @lru_cache(maxsize=None)
`def mel_filters(device, n_mels: int) -> torch.Tensor`
> load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
Allows decoupling librosa dependency; saved using:

    np.savez_compressed(
        "mel_filters.npz",
        mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
    )

### `log_mel_spectrogram`
`def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int=80, padding: int=0, device: Optional[Union[str, torch.device]]=None)`
> Compute the log-Mel spectrogram of

Parameters
----------
audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
    The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

n_mels: int
    The number of Mel-frequency filters, only 80 and 128 are supported

padding: int
    Number of zero samples to pad to the right

device: Optional[Union[str, torch.device]]
    If given, the audio tensor is moved to this device before STFT

Returns
-------
torch.Tensor, shape = (n_mels, n_frames)
    A Tensor that contains the Mel spectrogram
