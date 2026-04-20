#!/usr/bin/env python3
"""
一键运行 simulation 两步流程：
    1) generate_data_param.py  生成 meta.tsv 参数
    2) simulate_data_from_param.py  合成 clean/noisy 音频

使用方式:
    python gen_data/run_simulation.py --config gen_data/simulation_config.yaml --nj 4

说明:
    - 脚本会自动处理 espnet2 缺失问题（使用项目内置 mock 模块）
    - 会自动 patch detect_non_silence 的形状不匹配问题
    - 若 torchaudio.io 不存在，codec 增强将被跳过并打印警告
"""

import sys
from pathlib import Path
import subprocess
import argparse


def patch_detect_non_silence():
    """Monkey-patch detect_non_silence，避免 (1, T) 与 (T,) 布尔索引冲突。"""
    import numpy as np
    import espnet2.train.preprocessor as prep

    _original = prep.detect_non_silence

    def _patched(audio, threshold=0.01):
        result = _original(audio, threshold)
        # 保证返回形状与输入第一维一致；对单声道 (1, T) 扩展为 (1, T)
        if audio.ndim > 1 and result.ndim == 1:
            result = result[np.newaxis, :]
        return result

    prep.detect_non_silence = _patched


def run_generate_param(config_path):
    cmd = [
        sys.executable,
        "simulation/generate_data_param.py",
        "--config", str(config_path),
    ]
    print("[Step 1] Generating simulation parameters ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[Step 1] Done. meta.tsv generated.\n")


def run_simulate_audio(config_path, meta_tsv, nj, chunksize, highpass):
    cmd = [
        sys.executable,
        "simulation/simulate_data_from_param.py",
        "--config", str(config_path),
        "--meta_tsv", str(meta_tsv),
        "--nj", str(nj),
        "--chunksize", str(chunksize),
        "--highpass", str(highpass),
    ]
    print("[Step 2] Synthesizing audio ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[Step 2] Done. Audio synthesized.\n")


def main():
    parser = argparse.ArgumentParser(description="Run urgent2026 simulation pipeline")
    parser.add_argument(
        "--config", default="gen_data/simulation_config.yaml",
        help="Path to simulation YAML config"
    )
    parser.add_argument("--nj", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--chunksize", type=int, default=50, help="Chunk size for process_map")
    parser.add_argument("--highpass", type=str, default="True", help="Apply highpass filter")
    args = parser.parse_args()

    # 1. 确保项目根目录在 sys.path 中（使 espnet2 mock 生效）
    project_root = Path(__file__).parent.parent.resolve()
    sys.path.insert(0, str(project_root))

    # 2. Patch detect_non_silence shape bug
    patch_detect_non_silence()

    config_path = Path(args.config).resolve()
    meta_tsv = (Path("simulation_train/log") / "meta.tsv").resolve()

    # 3. Step 1: 生成参数
    run_generate_param(config_path)

    # 4. Step 2: 合成音频
    run_simulate_audio(config_path, meta_tsv, args.nj, args.chunksize, args.highpass)

    print("=" * 50)
    print("All done!")
    print(f"  Clean audio : simulation_train/clean/")
    print(f"  Noisy audio : simulation_train/noisy/")
    print(f"  Meta file   : {meta_tsv}")
    print("=" * 50)


if __name__ == "__main__":
    main()
