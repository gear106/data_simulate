#!/usr/bin/env python3
"""
离线生成 speech_scp、noise_scp、rir_scp 文件，格式与 dataloader/data_module.py 中 WaveInfo 的解析逻辑严格对齐。

SCP 格式说明：
- speech_scp: 3列  <utt_id> <spk_id> <path>
- noise_scp : 5列  <utt_id> <fs> <start_sample> <frames> <path>
- rir_scp   : 2列  <utt_id> <path>

使用示例：
    # 单独生成 speech.scp（3列）
    python generate_scp_offline.py --speech_dir /data/clean --out_dir ./scp

    # 单独生成 noise.scp（5列，需读取音频获取 fs/frames）
    python generate_scp_offline.py --noise_dir /data/noise --out_dir ./scp

    # 单独生成 rir.scp（2列）
    python generate_scp_offline.py --rir_dir /data/rir --out_dir ./scp

    # 同时生成三类
    python generate_scp_offline.py \
        --speech_dir /data/clean \
        --noise_dir /data/noise \
        --rir_dir /data/rir \
        --out_dir ./scp \
        --speech_rel_to /data

    # speech 路径输出为相对于 base_dir 的形式（与 config.yaml 中 speech_scp_base_dir 配合）
    python generate_scp_offline.py --speech_dir /voxbox/clean --speech_rel_to /voxbox --out_dir ./scp
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def scan_audio_files(directory):
    """递归扫描目录下的音频文件，以 stem 作为 utt_id"""
    files = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[Warning] directory not found: {directory}", file=sys.stderr)
        return files

    for ext in AUDIO_EXTS:
        for p in dir_path.rglob(f"*{ext}"):
            utt = p.stem
            if utt in files:
                print(f"[Warning] duplicate utt_id '{utt}' found:", file=sys.stderr)
                print(f"  existing: {files[utt]}", file=sys.stderr)
                print(f"  new:      {p}", file=sys.stderr)
            else:
                files[utt] = p.resolve()
    return files


def infer_spk(path, root):
    """尝试从相对目录结构推断 speaker，默认 unknown"""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
    except ValueError:
        pass
    return "unknown"


def get_audio_info(path):
    """读取音频文件，返回 (fs, frames)。出错时返回 None"""
    try:
        info = sf.info(path)
        return info.samplerate, info.frames
    except Exception as e:
        print(f"[Error] failed to read audio info: {path} ({e})", file=sys.stderr)
        return None


def format_path(path, rel_to=None):
    """根据 rel_to 返回相对路径或绝对路径字符串"""
    path = Path(path)
    if rel_to is not None:
        try:
            return str(path.relative_to(Path(rel_to).resolve()))
        except ValueError:
            pass
    return str(path)


def main():
    parser = argparse.ArgumentParser(
        description="离线生成 speech/noise/rir 三类 scp 文件"
    )
    parser.add_argument("--speech_dir", help="Clean 音频目录（生成 speech.scp，3列）")
    parser.add_argument("--noise_dir", help="Noise 音频目录（生成 noise.scp，5列）")
    parser.add_argument("--rir_dir", help="RIR 音频目录（生成 rir.scp，2列）")
    parser.add_argument(
        "--out_dir", required=True, help="SCP 文件输出目录"
    )
    parser.add_argument(
        "--speech_rel_to",
        default=None,
        help="speech 路径以此为基准输出相对路径（对应 config.yaml 中的 speech_scp_base_dir）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="读取 noise 音频信息的并行 workers 数（默认 8）",
    )
    args = parser.parse_args()

    if args.speech_dir is None and args.noise_dir is None and args.rir_dir is None:
        parser.error("至少需要提供 --speech_dir、--noise_dir、--rir_dir 中的一个")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- speech.scp (3列: utt spk path) ----------------
    if args.speech_dir:
        speech_files = scan_audio_files(args.speech_dir)
        speech_scp = out_dir / "speech.scp"
        root = Path(args.speech_dir).resolve()
        rel_base = Path(args.speech_rel_to).resolve() if args.speech_rel_to else None

        with open(speech_scp, "w", encoding="utf-8") as f:
            for utt in sorted(speech_files.keys()):
                path = speech_files[utt]
                spk = infer_spk(path, root)
                path_str = format_path(path, rel_to=rel_base)
                f.write(f"{utt} {spk} {path_str}\n")
        print(
            f"[Generated] {speech_scp} ({len(speech_files)} lines, 3列: utt spk path)"
        )

    # ---------------- noise.scp (5列: utt fs start frames path) ----------------
    if args.noise_dir:
        noise_files = scan_audio_files(args.noise_dir)
        noise_scp = out_dir / "noise.scp"

        items = sorted(noise_files.items())
        results = {}

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_utt = {
                executor.submit(get_audio_info, path): utt for utt, path in items
            }
            for future in as_completed(future_to_utt):
                utt = future_to_utt[future]
                info = future.result()
                if info is not None:
                    results[utt] = info
                else:
                    print(f"[Skip] {utt} due to read error", file=sys.stderr)

        with open(noise_scp, "w", encoding="utf-8") as f:
            for utt in sorted(results.keys()):
                fs, frames = results[utt]
                path_str = str(noise_files[utt])
                f.write(f"{utt} {fs} 0 {frames} {path_str}\n")
        print(
            f"[Generated] {noise_scp} ({len(results)} lines, 5列: utt fs start frames path)"
        )

    # ---------------- rir.scp (2列: utt path) ----------------
    if args.rir_dir:
        rir_files = scan_audio_files(args.rir_dir)
        rir_scp = out_dir / "rir.scp"

        with open(rir_scp, "w", encoding="utf-8") as f:
            for utt in sorted(rir_files.keys()):
                f.write(f"{utt} {rir_files[utt]}\n")
        print(f"[Generated] {rir_scp} ({len(rir_files)} lines, 2列: utt path)")

    print("Done.")


if __name__ == "__main__":
    main()
