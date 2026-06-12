# Analysis Report for audio.py

## Dependencies
- os
- functools.lru_cache
- subprocess.CalledProcessError
- subprocess.run
- typing.Optional
- typing.Union
- numpy
- torch
- torch.nn.functional

## Schemas
None

## API Contracts
- exact_div(x, y)
- load_audio(file, sr)
- pad_or_trim(array, length)
- mel_filters(device, n_mels)
- log_mel_spectrogram(audio, n_mels, padding, device)

## Configuration Variables
- SAMPLE_RATE
- N_FFT
- HOP_LENGTH
- CHUNK_LENGTH
- N_SAMPLES
- N_FRAMES
- N_SAMPLES_PER_TOKEN
- FRAMES_PER_SECOND
- TOKENS_PER_SECOND

## Assumptions & Notes
None

