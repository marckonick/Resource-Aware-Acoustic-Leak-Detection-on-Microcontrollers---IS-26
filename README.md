# Resource-Aware Acoustic Leak Detection on Microcontrollers: A Feature, Model, and Sequential-Decision Comparison for Compressed-Air Condition Monitoring
The official repository for the paper "Resource-Aware Acoustic Leak Detection on Microcontrollers: A Feature, Model, and Sequential-Decision Comparison for Compressed-Air Condition Monitoring" - IS 2026 

## Description 
Compressed air is among the most expensive energy carriers in industry, and undetected leaks account for a substantial share of wasted consumption. Acoustic leak detection in the audible range has been shown feasible with convolutional neural networks operating on high-resolution spectrograms, but such models assume studio-grade microphones and computational budgets far beyond those of low-cost embedded hardware. This work investigates whether reliable, continuous leak monitoring can instead be realized on microcontroller-class devices, with under 1 MB of flash and a few hundred KB of RAM. Using a publicly available compressed-air leakage dataset resampled to 16 kHz, we conduct a systematic comparison. First, we evaluate a hierarchy of feature representations of increasing cost, from simple energy and time-domain statistics through FFT spectral summaries, band-power descriptors, and MFCCs, up to compact log-mel patches. Second, we pair these features with lightweight classifiers, quantized tiny multilayer perceptrons, and small two-dimensional convolutional networks. Third, we add a temporal decision layer based on exponentially weighted moving averages, exploiting the persistent 
nature of leak emissions to accumulate evidence across frames rather than classifying isolated windows. Models are assessed under realistic strong background noise conditions. Crucially, detection quality is reported jointly with on-device resource cost, and inference time. Results characterize the accuracy-efficiency frontier and indicate that compact spectral features combined with temporally aggregated lightweight models can achive reliable condition moniotring for the considered application.

## Dataset info 

The dataset can be downloaded from the following repository - https://zenodo.org/records/7551606

## Files Description

Codes for Condition Monitoring

- Functions_FeatureExtraction.py          - ***
