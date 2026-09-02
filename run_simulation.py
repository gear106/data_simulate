# -*- coding: utf-8 -*-
"""
用 test_raw 的干净语音(ma_speech) + 纯噪声(ma_noise) 自己合成带噪语音，
用 Enny1991/beamformers 的频域 MVDR 做波束形成，保存增强结果并对真值(dp_speech, mic0)评估。

支持两种导向矢量估计方式（可同时跑、并排对比）：
  - practical（实用）：target=None，导向矢量从 "混合协方差 − 噪声协方差" 的主特征向量估计
                        —— 不偷看真值，是 MVDR 的真实水平；
  - oracle（神谕）  ：target=干净多通道语音，导向矢量直接用干净目标协方差估计
                        —— 偷看真值，是 MVDR 的理论上限。
两者差距越大，说明瓶颈在导向矢量估计；差距越小，说明瓶颈在 MVDR 方法本身。

合成遵循 RealMAN 原文 3.5 节：原始录制电平【直接相加】，不设 coeff（-0.8dB 是自然结果）。


# 两种模式都跑（推荐）
python run_mvdr_eval.py \
  --speech_pat '.../ma_speech/.../XXX_CH{ch}.flac' \
  --noise_pat '.../ma_noise/.../YYY_CH{ch}.flac' \
  --ref '.../dp_speech/.../XXX.flac' \
  --out out.wav --channels 0-31 --mode both
"""
import os
import argparse
import numpy as np
import soundfile as sf
from beamformers import beamformers

FS = 16000  # 统一重采样到 16k


# ---------------- IO ----------------
def load_mc(path_pattern: str, channels, target_sr=FS):
    """读多通道 flac，path_pattern 用 {ch} 占位通道号。返回 [C, T] float32"""
    wavs = []
    sr0 = None
    for c in channels:
        p = path_pattern.format(ch=c)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        w, sr = sf.read(p, dtype='float32')
        sr0 = sr
        wavs.append(w)
    wav = np.stack(wavs, axis=0)  # [C,T]
    if sr0 != target_sr:
        from scipy.signal import resample_poly
        wav = resample_poly(wav, up=target_sr, down=sr0, axis=-1).astype('float32')
    return wav


def load_mono(path, target_sr=FS):
    w, sr = sf.read(path, dtype='float32')
    if sr != target_sr:
        from scipy.signal import resample_poly
        w = resample_poly(w, up=target_sr, down=sr, axis=-1).astype('float32')
    return w


def match_len(noise, T):
    """把噪声对齐到长度 T：不够则循环拼接，够则随机截一段"""
    C, N = noise.shape
    if N >= T:
        s = np.random.randint(0, N - T + 1)
        return noise[:, s:s + T]
    reps = int(np.ceil(T / N))
    return np.tile(noise, (1, reps))[:, :T]


# ---------------- 指标 ----------------
def si_sdr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.dot(proj, proj) / (np.dot(noise, noise) + 1e-12) + 1e-12)


def snr(est, ref):
    n = min(len(est), len(ref))
    est, ref = est[:n].astype(np.float64), ref[:n].astype(np.float64)
    noise = est - ref
    return 10 * np.log10(np.dot(ref, ref) / (np.dot(noise, noise) + 1e-12) + 1e-12)


def try_pesq(ref, est, sr):
    try:
        from pesq import pesq as _pesq
        mode = 'wb' if sr == 16000 else 'nb'
        n = min(len(est), len(ref))
        return float(_pesq(sr, ref[:n], est[:n], mode))
    except Exception:
        return None  # pesq 未安装则跳过


# ---------------- 单次 MVDR ----------------
def run_mvdr(mix, noise, clean, mode):
    """
    mode='practical': target=None，导向矢量从 mix-noise 协方差估计（不偷看真值）
    mode='oracle'   : target=clean(多通道)，导向矢量用干净目标估计（上限）
    """
    if mode == 'oracle':
        if clean is None:
            raise ValueError('oracle 模式需要多通道干净语音 target（test_raw/ma_speech）')
        return beamformers.MVDR(mix, noise, target=clean, ref_mic=0).astype('float32')
    else:
        return beamformers.MVDR(mix, noise, target=None, ref_mic=0).astype('float32')


# ---------------- 单条处理 ----------------
def process_one(speech_pat, noise_pat, ref_path, channels, out_path, modes):
    """
    合成一条带噪语音，按 modes 列表分别跑 MVDR，返回 {mode: 指标dict}，并保存 wav。
    out_path 形如 xxx.wav，多模式时自动加后缀 xxx_practical.wav / xxx_oracle.wav。
    """
    clean = load_mc(speech_pat, channels)   # [C,T] 干净多通道语音（含混响）
    noise = load_mc(noise_pat, channels)    # [C,T] 纯噪声
    T = clean.shape[-1]
    noise = match_len(noise, T)

    # 直接相加，不调电平（与官方 test 构造一致）
    mix = clean + noise

    # 真值（mic0 直达声），两种模式共用
    ref = load_mono(ref_path)

    base, ext = os.path.splitext(out_path)
    results = {}
    for mode in modes:
        out = run_mvdr(mix, noise, clean, mode)
        op = f"{base}_{mode}{ext}" if len(modes) > 1 else out_path
        os.makedirs(os.path.dirname(op) or '.', exist_ok=True)
        sf.write(op, out, FS)
        results[mode] = {
            'SI_SDR': float(si_sdr(out, ref)),
            'SNR': float(snr(out, ref)),
            'PESQ': try_pesq(ref, out, FS),
            'wav': op,
        }
    return results


# ---------------- 主流程 ----------------
def parse_channels(s):
    if '-' in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speech_pat', required=True, help='test_raw ma_speech 路径模板，含 {ch}')
    ap.add_argument('--noise_pat', required=True, help='test_raw ma_noise 路径模板，含 {ch}（同场景）')
    ap.add_argument('--ref', required=True, help='test dp_speech 单通道真值（mic0）路径')
    ap.add_argument('--out', required=True, help='增强结果保存路径 .wav（多模式自动加后缀）')
    ap.add_argument('--channels', default='0-31', help='如 0-31 或 0,1,2,3')
    ap.add_argument('--mode', default='both',
                    choices=['practical', 'oracle', 'both'],
                    help='practical=混合-噪声协方差估计(实用)；oracle=干净目标估计(上限)；both=两者都跑')
    args = ap.parse_args()

    channels = parse_channels(args.channels)
    modes = ['practical', 'oracle'] if args.mode == 'both' else [args.mode]

    results = process_one(args.speech_pat, args.noise_pat, args.ref, channels, args.out, modes)

    print('\n================ MVDR 结果 ================')
    for mode, r in results.items():
        pesq = f"{r['PESQ']:.3f}" if r['PESQ'] is not None else 'N/A(未装pesq)'
        print(f"[{mode:9s}] SI-SDR={r['SI_SDR']:6.2f} dB | SNR={r['SNR']:6.2f} dB | PESQ={pesq} | wav -> {r['wav']}")
    if 'practical' in results and 'oracle' in results:
        gap = results['oracle']['SI_SDR'] - results['practical']['SI_SDR']
        print(f"\n导向矢量估计差距(oracle-practical) = {gap:.2f} dB "
              f"({'瓶颈在导向矢量估计' if gap > 1.5 else '瓶颈偏向 MVDR 方法本身'})")


if __name__ == '__main__':
    main()
