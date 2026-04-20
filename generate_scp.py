#!/usr/bin/env python3
"""
根据原始数据目录生成 simulation 所需的 .scp 与配套文件。

使用方式:
    python gen_data/generate_scp.py \
        --speech_dir "D:/code/data/VoiceBank/clean" \
        --noise_dir "D:/code/data/DEMAND" \
        --rir_dir "D:/code/data/RIR/rirs_noises/real_rirs_isotropic_noises" \
        --output_dir "data/train_sources" \
        --speech_fs 16000 \
        --noise_fs 16000 \
        --rir_fs 16000
"""

import argparse
from pathlib import Path


def make_scp(audio_dir, out_scp, fs, prefix="utt"):
    """遍历目录下所有 wav 文件，生成三列 scp: uid fs path"""
    audio_dir = Path(audio_dir)
    with open(out_scp, "w", encoding="utf-8") as f:
        i = 0
        for wav_path in sorted(audio_dir.rglob("*.wav")):
            uid = f"{prefix}_{i:05d}"
            f.write(f"{uid} {fs} {wav_path}\n")
            i += 1
    return i


def main():
    parser = argparse.ArgumentParser(
        description="Generate scp files for urgent2026 simulation"
    )
    parser.add_argument(
        "--speech_dir", required=True, help="干净语音根目录，会递归搜索 .wav"
    )
    parser.add_argument(
        "--noise_dir", required=True, help="噪声根目录，会递归搜索 .wav"
    )
    parser.add_argument(
        "--rir_dir", required=True, help="RIR 根目录，会递归搜索 .wav"
    )
    parser.add_argument(
        "--output_dir", default="data/train_sources", help="输出目录"
    )
    parser.add_argument("--speech_fs", type=int, default=16000, help="语音采样率")
    parser.add_argument("--noise_fs", type=int, default=16000, help="噪声采样率")
    parser.add_argument("--rir_fs", type=int, default=16000, help="RIR 采样率")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. speech_sources.scp
    n_speech = make_scp(
        args.speech_dir, out_dir / "speech_sources.scp", args.speech_fs, prefix="sp"
    )

    # 2. utt2spk (用 uid 本身作为 speaker id，若文件名含 speaker 信息可自行修改)
    with open(out_dir / "speech_sources.scp", "r") as f_in, open(
        out_dir / "utt2spk", "w"
    ) as f_out:
        for line in f_in:
            uid = line.strip().split()[0]
            f_out.write(f"{uid} {uid}\n")

    # 3. text (无转录时填 <not-available>)
    with open(out_dir / "speech_sources.scp", "r") as f_in, open(
        out_dir / "text", "w"
    ) as f_out:
        for line in f_in:
            uid = line.strip().split()[0]
            f_out.write(f"{uid} <not-available>\n")

    # 4. noise_scoures.scp
    n_noise = make_scp(
        args.noise_dir, out_dir / "noise_scoures.scp", args.noise_fs, prefix="noise"
    )

    # 5. wind_noise_scoures.scp (空文件，表示不使用 wind noise)
    with open(out_dir / "wind_noise_scoures.scp", "w"):
        pass

    # 6. rirs.scp
    n_rir = make_scp(args.rir_dir, out_dir / "rirs.scp", args.rir_fs, prefix="rir")

    # 7. source_length.scp (使用 soundfile 快速读取帧数)
    import soundfile as sf

    with open(out_dir / "speech_sources.scp", "r") as f_in, open(
        out_dir / "source_length.scp", "w"
    ) as f_out:
        for line in f_in:
            uid, fs, path = line.strip().split()
            info = sf.info(path)
            f_out.write(f"{uid} {info.frames}\n")

    print("=" * 50)
    print(f"Speech  files: {n_speech}")
    print(f"Noise   files: {n_noise}")
    print(f"RIR     files: {n_rir}")
    print(f"Output  dir  : {out_dir.absolute()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
