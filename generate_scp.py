import argparse
from pathlib import Path

AUDIO_EXTS = {'.wav', '.flac', '.mp3'}


def get_audio_files(directory):
    """递归获取目录下所有音频文件，以 stem 为 utt_id"""
    files = {}
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[Warning] directory not found: {directory}")
        return files
    for ext in AUDIO_EXTS:
        for p in dir_path.rglob(f'*{ext}'):
            utt = p.stem
            if utt in files:
                print(f"[Warning] duplicate utt_id '{utt}' found:")
                print(f"  existing: {files[utt]}")
                print(f"  new:      {p}")
            else:
                files[utt] = p.resolve()
    return files


def infer_spk(path, root):
    """尝试从目录的相对路径推断 speaker，默认 unknown"""
    try:
        rel = path.relative_to(root)
        parts = rel.parts
        if len(parts) > 1:
            return parts[0]
    except ValueError:
        pass
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description="从 clean/noisy/rir 目录生成 SCP 文件，支持单独指定任意一种或多种，自动检查 clean/noisy 配对"
    )
    parser.add_argument('--clean_dir',
                        help='Clean 音频目录（生成 speech.scp / clean.scp）')
    parser.add_argument('--noisy_dir',
                        help='Noisy 音频目录（生成 mix.scp），与 clean 同时提供时会做配对检查')
    parser.add_argument('--rir_dir',
                        help='RIR 音频目录（生成 rir.scp）')
    parser.add_argument('--out_dir', required=True,
                        help='SCP 文件输出目录')
    args = parser.parse_args()

    if args.clean_dir is None and args.noisy_dir is None and args.rir_dir is None:
        parser.error("至少需要提供 --clean_dir、--noisy_dir、--rir_dir 中的一个")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按需扫描
    clean_files = get_audio_files(args.clean_dir) if args.clean_dir else {}
    noisy_files = get_audio_files(args.noisy_dir) if args.noisy_dir else {}
    rir_files = get_audio_files(args.rir_dir) if args.rir_dir else {}

    # 配对检查（仅在 clean 和 noisy 都提供时）
    if clean_files and noisy_files:
        common_utts = sorted(set(clean_files.keys()) & set(noisy_files.keys()))
        clean_only = sorted(set(clean_files.keys()) - set(noisy_files.keys()))
        noisy_only = sorted(set(noisy_files.keys()) - set(clean_files.keys()))

        print("=" * 60)
        print(f"Clean 音频总数 : {len(clean_files)}")
        print(f"Noisy 音频总数 : {len(noisy_files)}")
        print(f"Clean&Noisy 配对成功 : {len(common_utts)}")
        if clean_only:
            print(f"Clean 未配对   : {len(clean_only)} (示例: {clean_only[:5]})")
        if noisy_only:
            print(f"Noisy 未配对   : {len(noisy_only)} (示例: {noisy_only[:5]})")
        print("=" * 60)
    else:
        common_utts = None
        print("=" * 60)
        if clean_files:
            print(f"Clean 音频总数 : {len(clean_files)} (未提供 noisy，不做配对)")
        if noisy_files:
            print(f"Noisy 音频总数 : {len(noisy_files)} (未提供 clean，不做配对)")
        if rir_files:
            print(f"RIR   音频总数 : {len(rir_files)}")
        print("=" * 60)

    # 1) speech.scp (clean, 3列: utt spk path)
    if clean_files:
        speech_scp = out_dir / 'speech.scp'
        root = Path(args.clean_dir)
        with open(speech_scp, 'w', encoding='utf-8') as f:
            for utt in sorted(clean_files.keys()):
                path = clean_files[utt]
                spk = infer_spk(path, root)
                f.write(f"{utt} {spk} {path}\n")
        print(f"[Generated] {speech_scp} ({len(clean_files)} lines, 3列: utt spk path)")

    # 2) clean.scp (clean, 3列: utt spk path)
    if clean_files:
        clean_scp = out_dir / 'clean.scp'
        root = Path(args.clean_dir)
        utts_to_write = common_utts if common_utts is not None else sorted(clean_files.keys())
        with open(clean_scp, 'w', encoding='utf-8') as f:
            for utt in utts_to_write:
                path = clean_files[utt]
                spk = infer_spk(path, root)
                f.write(f"{utt} {spk} {path}\n")
        print(f"[Generated] {clean_scp} ({len(utts_to_write)} lines, 3列: utt spk path)")

    # 3) mix.scp (noisy, 3列: utt spk path)
    if noisy_files:
        mix_scp = out_dir / 'mix.scp'
        root = Path(args.noisy_dir)
        utts_to_write = common_utts if common_utts is not None else sorted(noisy_files.keys())
        with open(mix_scp, 'w', encoding='utf-8') as f:
            for utt in utts_to_write:
                path = noisy_files[utt]
                spk = infer_spk(path, root)
                f.write(f"{utt} {spk} {path}\n")
        print(f"[Generated] {mix_scp} ({len(utts_to_write)} lines, 3列: utt spk path)")

    # 4) rir.scp (rir, 2列: utt path)
    if rir_files:
        rir_scp = out_dir / 'rir.scp'
        with open(rir_scp, 'w', encoding='utf-8') as f:
            for utt in sorted(rir_files.keys()):
                f.write(f"{utt} {rir_files[utt]}\n")
        print(f"[Generated] {rir_scp} ({len(rir_files)} lines, 2列: utt path)")

    print("Done.")


if __name__ == '__main__':
    main()
