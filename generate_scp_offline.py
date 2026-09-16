import numpy as np, soundfile as sf
from scipy.io import loadmat
from scipy.signal import resample_poly
import sys; sys.path.insert(0, '~/AmbiDrop')
from ambidrop.asm import encode_ambisonics

mic48, fs = sf.read('ex_1/p.wav')           # (T, 9)
mic16 = resample_poly(mic48.T, 1, 3, axis=1) # 48k->16k

V = loadmat('RealMAN9ch (simulated).mat')['V']
grid = loadmat('Lebvedev2702.mat')
anm, _ = encode_ambisonics(mic16, V, sh_order=2,
                           th=grid['th'].squeeze(), ph=grid['ph'].squeeze())
# anm: (9, T) —— 第0通道是A00
for i in range(9):
    sf.write(f'ch{i}.wav', anm[i] / (np.abs(anm[i]).max() + 1e-8), 16000)
