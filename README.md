# Resource-Aware Acoustic Leak Detection on Microcontrollers: A Feature, Model, and Sequential-Decision Comparison for Compressed-Air Condition Monitoring
The official repository for the paper "Resource-Aware Acoustic Leak Detection on Microcontrollers: A Feature, Model, and Sequential-Decision Comparison for Compressed-Air Condition Monitoring" - IS 2026 

## Description 
Compressed air is among the most expensive energy carriers in industry, and undetected leaks account for a substantial share of wasted consumption. Acoustic leak detection in the audible range has been shown feasible with convolutional neural networks operating on high-resolution spectrograms, but such models assume studio-grade microphones and computational budgets far beyond those of low-cost embedded hardware. This work investigates whether reliable, continuous leak monitoring can instead be realized on microcontroller-class devices, with under 1 MB of flash and a few hundred KB of RAM. Using a publicly available compressed-air leakage dataset resampled to 16 kHz, we conduct a systematic comparison. First, we evaluate a hierarchy of feature representations of increasing cost, from simple energy and time-domain statistics through FFT spectral summaries, band-power descriptors, and MFCCs, up to compact log-mel patches. Second, we pair these features with lightweight classifiers, quantized tiny multilayer perceptrons, and small two-dimensional convolutional networks. Third, we add a temporal decision layer based on exponentially weighted moving averages, exploiting the persistent 
nature of leak emissions to accumulate evidence across frames rather than classifying isolated windows. Models are assessed under realistic strong background noise conditions. Crucially, detection quality is reported jointly with on-device resource cost, and inference time. Results characterize the accuracy-efficiency frontier and indicate that compact spectral features combined with temporally aggregated lightweight models can achive reliable condition moniotring for the considered application.

## Dataset info 

The dataset can be downloaded from the following repository - https://zenodo.org/records/7551606

All experiments use the IDMT compressed-air leakage dataset \cite{Grollmisch2019}. 
Recordings were made on a Festo Didactic pneumatic rig in a laboratory. Leakage was generated
by a choke vent whose aperture was set with a knurled screw, advanced from
zero to nine turns across $16$ discrete states (full rotations up to three
turns, half rotations thereafter), while each state was recorded for $30$\,s. Three
leakage types are present: a \emph{vent leak} at the nominal $6$\,bar, a
quieter \emph{vent low} at $5$\,bar, and a \emph{tube leak} from damaged
tubing. To emulate industrial conditions, each type was recorded under
laboratory noise and under workshop and hydraulic-press noise replayed
through loudspeakers at two levels, giving $15$ configurations across three
independent sessions. Four Earthworks M30 microphones ($3$\,Hz--$30$\,kHz)
recorded in parallel, positioned at $20$\,cm/$90^{\circ}$, $2$\,m/$90^{\circ}$
and $20$\,cm/$30^{\circ}$ from the source, with a fourth at an omnidirectional
reference position in the room.

## Files Description

Python code for development of the condition monitoring systems:

- config_feat_extraction.yaml            - Configuration file for feature extraction script - Main_ExtractFeatures_.py.
- config_classification.yaml             - Configuration file for classifier training script -  Master_ClassificationScript_vents.py.
- config_testing.yaml                    - Configuration file for computing result metrics script - Master_TestingScript_vents.py.
- FeatureExtractrionFunctions.py         - Functions for performing feature extraction spectrograms. Check the path to the data folder -  CNC_Machining-main which should be dowloaded first (see the previous section).  
- Main_ExtractFeatures_.py               - Main script for feature extraction.
- Master_ClassificationScript_vents.py   - Main script for classifier training.
- Master_TestingScript_vents.py          - Main script for computing final test metrics.

Arduino code used for implementation on Arduino 33 BLE SENSE microcontroller. Consists of 4 folders:  

- vent_leak_full_spectral_summ_arduino_nano_33 - CM1 system
- vent_leak_full_bandpower_arduino_nano_33 - CM2 system
- vent_leak_full_mfcc_arduino_nano_33 - CM3 system
- vent_leak_full_logmel_arduino_nano_33 - CM4 system
