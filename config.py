"""
config.py — control panel for the Channel Estimator.
"""

import numpy as np
# PHYSICS & DATA GENERATION   

# Link & system
L            = 1000.0    # m, link distance 
D_TX         = 0.02      # m                              
D_R          = 0.40      # m     
THETA_DIV    = 1e-3      # rad, half-angle divergence     
THETA_JITTER = 30e-6     # rad                            
TX_POWER     = 0.01      # W                              
ETA          = 9.3       # A/W responsivity               
NEP          = 5e-14     # W/√Hz  
BANDWIDTH    = 1e9       # Hz     

# Weather: FIXED CLEAR (attenuation only; turbulence strength set separately)
CLEAR_ATTEN_DB_KM = 0.2  # dB/km                          [Trichili Table 1]

# Turbulence: h_t generator
# (alpha, beta) sourced by regime from Trichili Fig. 7.
TURBULENCE_REGIMES = {
    "weak":     {"alpha": 11.6, "beta": 10.1},  
    "moderate": {"alpha": 4.0,  "beta": 1.9},    
    "strong":   {"alpha": 4.2,  "beta": 1.4},   
}

# Temporal correlation (coherence times, seconds)
# Within Trichili's FSO range 100 µs – 10 ms
TAU_SLOW = 10e-3    
TAU_FAST = 1e-3       

# Sampling & length
DT        = 1e-4     # s/step 
N_SAMPLES = 100_000  
DELAY_SECONDS = 1e-3     # s; ACM switching delay, placeholder

HORIZON_K = int(DELAY_SECONDS / DT)   #steps ahead

# MODEL PARAMS  
# placeholders; settled when models.py / train.py are built.

WINDOW_N    = 100    # input window ≈ TAU_SLOW/DT    

HIDDEN_SIZE = 64     
NUM_LAYERS  = 1      

LR          = 1e-3   # Adam learning rate
NUM_EPOCHS  = 50                                          
BATCH_SIZE  = 64     
TRAIN_FRAC  = 0.8    

# ACM & EVALUATION

TARGET_BER = 1e-3    # tune

# Mode table 
# placeholder SNR, selection logic lives in acm.py
MODE_TABLE = {
    "OOK":   {"bits_per_sym": 1, "snr_req_db": 13.5},   
    "4-PAM": {"bits_per_sym": 2, "snr_req_db": 20.0},   
    "8-PAM": {"bits_per_sym": 3, "snr_req_db": 26.0},   
}

SWITCHING_MARGIN_DB = 1.0   
EBN0_RANGE_DB = np.arange(0, 35, 2.5)   # SE-curve axis