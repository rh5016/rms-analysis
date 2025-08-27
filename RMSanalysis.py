import musdb
import numpy as np
import pandas as pd
import torch
from openunmix.predict import separate
from IPython.display import Audio, display
import librosa
import os

MUSDB_PATH = "/Users/rithuhegde/Desktop/musdb18hq"
block_size = 4096
silence_db_threshold = -60
targets = ['vocals', 'drums', 'bass', 'other']
sr = 44100

def add_silence_middle(audio, sr, duration):
    n_silence = int(duration * sr)
    audio_length = len(audio)
    
    # calculate middle position
    mid_point = audio_length // 2
    
    # split audio into two halves
    first_half = audio[:mid_point]
    second_half = audio[mid_point:]
    
    # Create silence
    silence = np.zeros((n_silence, audio.shape[1]), dtype=audio.dtype)
    
    # first half + silence + second half
    return np.concatenate([first_half, silence, second_half], axis=0)

def compute_rms_blocks(audio, block_size):
    rms = []
    mono = np.mean(audio, axis=1)  # stereo→mono
    for i in range(0, len(mono) - block_size + 1, block_size):
        blk = mono[i:i+block_size]
        rms.append(np.sqrt(np.mean(blk**2)))
    return np.array(rms)

def compute_snr(target, estimate):
    e = 1e-10
    S = np.sum(target**2)
    N = np.sum((target - estimate)**2)
    snr = 10 * np.log10((S+e)/(N+e))
    return float(snr)

music = musdb.DB(root=MUSDB_PATH, subsets='test', is_wav=True)
rows = []

for track in music.tracks:
    print(f"\nSeparating track: {track.name}")
    orig_stereo = track.audio
    
    for sec in (0, 30, 60, 90, 120):
        print(f"adding {sec}s silence in middle...")
        
        # pad original track with silence in middle
        padded = add_silence_middle(orig_stereo, sr, sec)
        
        # compute RMS on padded audio
        rms_vals = compute_rms_blocks(padded, block_size)
        rms_db = 20 * np.log10(rms_vals + 1e-10) #formula for converting to dB
        percent_silence = np.mean(rms_db < silence_db_threshold) * 100
        
        # separate the padded audio
        padded_tensor = torch.tensor(padded.T, dtype=torch.float32) #trasnpose to channles, samples
        with torch.no_grad():
            estimates = separate(padded_tensor, rate=sr, model_str_or_path="umxhq")
        
        for target in targets:
            #ground truth
            gt_source = track.sources[target].audio
            gt_source_padded = add_silence_middle(gt_source, sr, sec)
            
            # same shape check
            tgt_arr = gt_source_padded.T
            est_arr = estimates[target]
            
            if isinstance(est_arr, torch.Tensor):
                est_arr = est_arr.cpu().numpy()
            if est_arr.ndim == 3:
                est_arr = est_arr.squeeze(0)
            
            # dimension mismatch check
            L = min(tgt_arr.shape[1], est_arr.shape[1])
            
            if L > 0:
                snr = compute_snr(tgt_arr[:, :L], est_arr[:, :L])
                
                rows.append({
                    "Track": track.name,
                    "Source": target,
                    "Silence (s)": sec,
                    "SNR (dB)": snr,
                    "% Silence": percent_silence
                })
                
                print(f"      {target}: SNR = {snr:.3f} dB")


if rows:
    df = pd.DataFrame(rows)
    df_wide = df.pivot(index=['Track', 'Source'],
                       columns='Silence (s)',
                       values='SNR (dB)')

    df_wide = df_wide.rename_axis(None, axis=1).reset_index()

    df.to_csv("openunmix_snr_analysis_middle_silence.csv", index=False)
    print("results saved to openunmix_snr_analysis_middle_silence.csv")
else:
    print("No results generated. Check your dataset structure.")

print("done")