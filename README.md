# README: Modulation and GLRT Classification

## Overview

This repository contains Python scripts for generating and classifying modulated signals using M-PSK and M-QAM schemes. The classification is performed using the **Generalized Likelihood Ratio Test (GLRT)** in the presence of fading and noise.

## File Descriptions

### `modulation.py`

This script handles the generation of modulated signals:

- **`generate_mpsk_samples(M, N)`**: Generates M-PSK modulated samples.
- **`generate_mqam_samples(M, N)`**: Generates M-QAM modulated samples (supports 4-QAM, 8-QAM, 16-QAM, 32-QAM, and 64-QAM).
- **Constellation Plot Functions**: Allows visualization of modulation schemes.
- **Predefined Modulation Dictionaries**: Contains predefined alphabets for PSK and QAM modulations.

### `glrt.py`

This script performs modulation classification using the GLRT algorithm:

- **`apply_fading_channel(signal)`**: Simulates a fading channel.
- **`add_awgn(signal, snr_db)`**: Adds AWGN noise to a signal.
- **`Log_Joint_Likelihood(...)`**: Computes the likelihood function for a given modulation scheme.
- **`glrt_classification(...)`**: Classifies received symbols into a modulation scheme.
- **`test_GLRT(...)`**: Simulates and tests classification accuracy.
- **`taux_erreur(...)`**: Computes classification error rates over multiple SNR levels.

## Usage

### Generating Modulated Signals

To generate and visualize a modulation scheme, run:

```python
from modulation import plot_mod, ALPHABET
plot_mod("16QAM", ALPHABET)
```

### Running GLRT Classification

To classify a received signal:

```python
from glrt import test_GLRT
result = test_GLRT("16QAM", 500, 16, ALPHABET, plots=True)
print("Classification Result:", result)
```

### Error Rate Evaluation

To compute classification performance:

```python
from glrt import taux_erreur, MODLIST
SNRs = list(range(-20, 21, 2))
taux_erreur(MODLIST, SNRs, 500, 100, ALPHABET)
```

> You will want to modify functions  `taux_erreur` and `test_glrt` to take into acount fading and noise when testing

## Dependencies

- `numpy`
- `matplotlib`

Install missing dependencies using:

```bash
pip install numpy matplotlib
```

## License

This project is open-source and free to use.

