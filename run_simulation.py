import numpy as np
import librosa
from scipy.signal import stft

def estimate_snr_practical(noisy_speech_path, noise_only_path, 
                           frame_length=512, hop_length=256):
    """
    基于静音段参考的SNR估计
    输入: 带噪语音wav, 纯噪声wav
    输出: 全局SNR, 分段SNR分布, 频域SNR
    """
    # 加载（强制同采样率）
    s_n, sr = librosa.load(noisy_speech_path, sr=None)
    n_only, _ = librosa.load(noise_only_path, sr=sr)
    
    # 确保噪声样本足够，不够则循环拼接
    if len(n_only) < len(s_n):
        repeats = int(np.ceil(len(s_n) / len(n_only)))
        n_only = np.tile(n_only, repeats)[:len(s_n)]
    
    # ========== 方法1: 全局能量SNR（对标设备规格书）==========
    P_noise = np.mean(n_only ** 2)
    P_total = np.mean(s_n ** 2)
    P_signal = max(P_total - P_noise, 1e-12)
    
    snr_global = 10 * np.log10(P_signal / P_noise)
    
    # ========== 方法2: 分段SNR（更鲁棒，反映时域波动）==========
    # 分帧计算
    frames_total = librosa.util.frame(s_n, frame_length=frame_length, hop_length=hop_length)
    frames_noise = librosa.util.frame(n_only[:len(s_n)], frame_length=frame_length, hop_length=hop_length)
    
    # 每帧功率
    P_total_frames = np.mean(frames_total ** 2, axis=0)
    P_noise_frames = np.mean(frames_noise ** 2, axis=0)
    P_signal_frames = np.maximum(P_total_frames - P_noise_frames, 1e-12)
    
    snr_frames = 10 * np.log10(P_signal_frames / P_noise_frames)
    snr_median = np.median(snr_frames)  # 中位数比均值更抗异常帧
    
    # ========== 方法3: 频域SNR（看哪些频段被污染严重）==========
    _, _, Z_s = stft(s_n, fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)
    _, _, Z_n = stft(n_only[:len(s_n)], fs=sr, nperseg=frame_length, noverlap=frame_length-hop_length)
    
    # 功率谱
    P_s = np.mean(np.abs(Z_s) ** 2, axis=1)
    P_n = np.mean(np.abs(Z_n) ** 2, axis=1)
    P_sig = np.maximum(P_s - P_n, 1e-12)
    
    snr_freq = 10 * np.log10(P_sig / P_n)
    freqs = np.fft.rfftfreq(frame_length, 1/sr)
    
    return {
        'snr_global_db': float(snr_global),
        'snr_median_db': float(snr_median),
        'snr_frames_db': snr_frames,           # 时域分布
        'snr_per_freq_db': snr_freq,           # 频域分布
        'freqs': freqs,
        'sr': sr
    }
result = estimate_snr_practical("noisy_speech.wav", "silence.wav")

print(f"全局SNR: {result['snr_global_db']:.1f} dB")
print(f"中位数分段SNR: {result['snr_median_db']:.1f} dB")

# 看频域细节：哪些频段噪声大
for f, snr in zip(result['freqs'][::10], result['snr_per_freq_db'][::10]):
    if f <= 8000:  # 只看8kHz以内
        print(f"{f:.0f}Hz: {snr:.1f} dB")
