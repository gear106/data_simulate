#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realman_ambidrop_pipeline.py
============================
用 AmbiDrop 官方仓库 (github.com/mikitt012/AmbiDrop) 的默认权重推理 RealMAN 测试集。

流程:
    [阶段1] 生成 RealMAN 9通道阵列的 steering 矩阵 (.mat)   --> --step steering
    [阶段2] 把 RealMAN 原始数据转成 AmbiDrop 评测目录结构   --> --step data
    [阶段3] 打印/执行 run_Real_World.py 推理命令            --> --step run

用法示例:
    python realman_ambidrop_pipeline.py --step all \\
        --ambidrop-repo ~/AmbiDrop \\
        --realman-root ~/datasets/RealMAN \\
        --out-root ~/AmbiDrop/datasets/realman_eval

依赖: numpy scipy soundfile (阶段2还需要 resample: scipy.signal.resample_poly)
"""

import argparse
import os
import sys
import glob
import subprocess

import numpy as np

# =====================================================================
# 用户配置区 —— 按你的实际环境修改
# =====================================================================

# RealMAN 9通道子阵几何 (单位: 米), 顺序 = p.wav 通道顺序
# ⚠️ 对角麦克风坐标务必与 RealMAN 数据集附带的几何文件核对!
#    mic0 在原点; mic1/3/5/7 在 ±x/±y 轴 3cm; mic2/4/6/8 在对角 3cm 半径处
MIC_POS = np.array([
    [ 0.000,  0.000, 0.0],   # 0  center
    [ 0.030,  0.000, 0.0],   # 1  +x
    [ 0.0212, 0.0212, 0.0],  # 2  diag NE  (3cm/sqrt(2), 以数据集几何文件为准)
    [ 0.000,  0.030, 0.0],   # 3  +y
    [-0.0212, 0.0212, 0.0],  # 4  diag NW
    [-0.030,  0.000, 0.0],   # 5  -x
    [-0.0212,-0.0212, 0.0],  # 6  diag SW
    [ 0.000, -0.030, 0.0],   # 7  -y
    [ 0.0212,-0.0212, 0.0],  # 8  diag SE
])

MIC_POS_32 = np.array([
    # 中心
    [ 0.000,  0.000,  0.000],   # 0
    # x 轴
    [ 0.030,  0.000,  0.000],   # 1
    [ 0.060,  0.000,  0.000],   # 9
    [ 0.090,  0.000,  0.000],   # 17
    [ 0.120,  0.000,  0.000],   # 26
    [ 0.150,  0.000,  0.000],   # 27
    [-0.030,  0.000,  0.000],   # 5
    [-0.060,  0.000,  0.000],   # 13
    [-0.090,  0.000,  0.000],   # 21
    [-0.120,  0.000,  0.000],   # 25
    # y 轴
    [ 0.000,  0.030,  0.000],   # 3
    [ 0.000,  0.060,  0.000],   # 11
    [ 0.000,  0.090,  0.000],   # 19
    [ 0.000, -0.030,  0.000],   # 7
    [ 0.000, -0.060,  0.000],   # 15
    [ 0.000, -0.090,  0.000],   # 23
    # 对角臂 (3cm间距沿臂 → 半径 3/6/9cm, 45°)
    [ 0.0212,  0.0212, 0.000],  # 2   [0.0424, 0.0424]  # 10  [0.0636, 0.0636]  # 18
    [-0.0212,  0.0212, 0.000],  # 4   [-0.0424, 0.0424] # 12  [-0.0636, 0.0636] # 20
    [-0.0212, -0.0212, 0.000],  # 6   [-0.0424,-0.0424] # 14  [-0.0636,-0.0636] # 22
    [ 0.0212, -0.0212, 0.000],  # 8   [0.0424,-0.0424]  # 16  [0.0636,-0.0636] # 24
    # z 轴
    [ 0.000,  0.000,  0.045],   # 28
    [ 0.000,  0.000,  0.090],   # 29
    [ 0.000,  0.000, -0.045],   # 30
    [ 0.000,  0.000, -0.090],   # 31
])
# 对角臂坐标请按之前说的 TDOA 方法核对一遍

FS_TARGET = 16000      # AmbiDrop 模型采样率
NFFT = 512             # 必须匹配 ambidrop/asm.py apply_asm_filters 的 filt_samp=512
SPEED_OF_SOUND = 343.0

# RealMAN 原始录音采样率 (论文: 48kHz)
FS_REALMAN = 48000

# =====================================================================
# 阶段1: 生成 steering 矩阵
# =====================================================================

def build_steering(ambidrop_repo: str):
    from scipy.io import loadmat, savemat

    grid_path = os.path.join(ambidrop_repo, "utils", "Lebvedev2702.mat")
    assert os.path.exists(grid_path), f"找不到求积网格: {grid_path}"

    grid = loadmat(grid_path)
    th, ph = grid["th"].squeeze(), grid["ph"].squeeze()   # 极角/方位角, Q=2702
    Q = len(th)
    u = np.stack([np.sin(th) * np.cos(ph),
                  np.sin(th) * np.sin(ph),
                  np.cos(th)], axis=1)                     # (Q, 3) 方向单位向量

    M = MIC_POS.shape[0]
    F_pos = NFFT // 2 + 1                                   # 257
    freqs = np.arange(F_pos) * FS_TARGET / NFFT

    V = np.zeros((M, F_pos, Q), dtype=complex)
    for f in range(1, F_pos):                               # 跳过 DC (steering 秩1退化)
        k = 2 * np.pi * freqs[f] / SPEED_OF_SOUND
        V[:, f, :] = np.exp(-1j * k * (MIC_POS @ u.T))

    out_dir = os.path.join(ambidrop_repo, "utils", "steering")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "RealMAN9ch (simulated).mat")
    savemat(out_path, {"V": V})
    print(f"[steering] 已保存: {out_path}  shape={V.shape} (期望 (9, 257, 2702))")
    print("[steering] 若推理结果方位镜像, 将 exp(-1j*...) 改为 exp(+1j*...) 重新生成本文件")
    return out_path

# =====================================================================
# 阶段2: RealMAN -> AmbiDrop ex_* 目录结构
# =====================================================================
#
# 目标结构:
#   <out_root>/<scenario>/ex_<N>/{p.wav, s.wav, best_shift.txt}
#     p.wav : (T, 9) 48kHz 多通道含噪语音
#     s.wav : (T',)  16kHz 单通道干净目标 (mic0 直达声降采样)
#
# RealMAN 原始数据的两种常见形态, 按你的实际情况实现 fetch_one():
#   A) 每条utterance一个多通道wav (T, 32) -> 取9列
#   B) 每utterance每mic一个文件 xxx_micXX.wav -> 读9个文件堆叠

def find_realman_utterances(realman_root: str):
    """返回 utterance 列表, 每个元素是 (mix_wav_path 或 [mic文件列表], direct_path, scenario_tag)"""
    # TODO: 按你下载的 RealMAN 目录结构调整下面两行
    speech_dir = os.path.join(realman_root, "test", "speech")        # 多通道语音目录
    direct_dir = os.path.join(realman_root, "test", "direct_path")   # mic0 直达声目标目录
    mix_files = sorted(glob.glob(os.path.join(speech_dir, "**", "*.wav"), recursive=True))
    entries = []
    for mix in mix_files:
        utt_id = os.path.splitext(os.path.basename(mix))[0]
        direct = os.path.join(direct_dir, utt_id + ".wav")
        scenario = os.path.basename(os.path.dirname(mix)) or "realman"
        if os.path.exists(direct):
            entries.append((mix, direct, scenario))
        else:
            print(f"[warn] 缺直达声目标, 跳过: {utt_id}")
    assert entries, "没有找到任何 utterance, 请检查 find_realman_utterances() 的路径配置"
    return entries

def prepare_data(realman_root: str, out_root: str, limit: int = None):
    import soundfile as sf
    from scipy.signal import resample_poly

    entries = find_realman_utterances(realman_root)
    if limit:
        entries = entries[:limit]

    n_written = 0
    for mix_path, direct_path, scenario in entries:
        x, fs = sf.read(mix_path)                 # 期望 (T, C) 或 (T,)
        if x.ndim == 1:
            raise ValueError(f"{mix_path} 是单通道文件, 请检查是否选对多通道语音")
        assert fs == FS_REALMAN, f"采样率 {fs} != {FS_REALMAN}"
        x9 = x[:, :9]                             # 取前9通道 = mic 0~8, 顺序须与 MIC_POS 一致

        s48, fs_s = sf.read(direct_path)          # mic0 直达声 (单通道)
        assert fs_s == FS_REALMAN
        s16 = resample_poly(s48, up=1, down=FS_REALMAN // FS_TARGET).astype(np.float32)

        ex_dir = os.path.join(out_root, scenario, f"ex_{n_written + 1}")
        os.makedirs(ex_dir, exist_ok=True)
        sf.write(os.path.join(ex_dir, "p.wav"), x9, FS_REALMAN, subtype="FLOAT")
        sf.write(os.path.join(ex_dir, "s.wav"), s16, FS_TARGET, subtype="FLOAT")
        with open(os.path.join(ex_dir, "best_shift.txt"), "w") as f:
            f.write("0")                          # 占位即可, 主流程用互相关重新对齐

        n_written += 1
        if n_written % 50 == 0:
            print(f"[data] 已处理 {n_written}/{len(entries)}")

    print(f"[data] 完成: {n_written} 条 -> {out_root}")
    print("[data] 提示: s.wav (直达声) 与模型输出 A00 差 1/sqrt(4π) 幅度, 但指标均幅度不变, 无需处理")

# =====================================================================
# 阶段3: 推理
# =====================================================================

def run_inference(ambidrop_repo: str, out_root: str, steering_path: str,
                  scenarios=None, regularization="tikhonov", dry_run=True):
    cmd = [
        sys.executable, os.path.join(ambidrop_repo, "run_Real_World.py"),
        "--atf", "simulated",
        "--steering-path", steering_path,
        "--grid-path", os.path.join(ambidrop_repo, "utils", "Lebvedev2702.mat"),
        "--aria-data-dir", out_root,
        "--ref-mic", "1",
        "--regularization", regularization,
        "--output-csv", os.path.join(out_root, "results.csv"),
    ]
    if scenarios:
        cmd += ["--scenarios"] + list(scenarios)
    print("[run] 执行命令:")
    print("      " + " ".join(cmd))
    if dry_run:
        print("[run] dry-run 模式, 去掉 --dry-run 实际执行")
        return
    subprocess.run(cmd, cwd=ambidrop_repo, check=True)

# =====================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["steering", "data", "run", "all"], default="all")
    p.add_argument("--ambidrop-repo", required=True, help="AmbiDrop 仓库根目录")
    p.add_argument("--realman-root", default=None, help="RealMAN 数据集根目录 (阶段2需要)")
    p.add_argument("--out-root", default=None, help="评测数据输出目录 (阶段2/3)")
    p.add_argument("--limit", type=int, default=None, help="只处理前N条(调试用)")
    p.add_argument("--regularization", choices=["tikhonov", "svd"], default="tikhonov")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    steering_path = os.path.join(args.ambidrop_repo, "utils", "steering",
                                 "RealMAN9ch (simulated).mat")

    if args.step in ("steering", "all"):
        build_steering(args.ambidrop_repo)

    if args.step in ("data", "all"):
        assert args.realman_root and args.out_root, "阶段2需要 --realman-root 和 --out-root"
        prepare_data(args.realman_root, args.out_root, limit=args.limit)

    if args.step in ("run", "all"):
        out_root = args.out_root or os.path.join(args.ambidrop_repo, "datasets", "realman_eval")
        run_inference(args.ambidrop_repo, out_root, steering_path,
                      scenarios=args.scenarios,
                      regularization=args.regularization,
                      dry_run=args.dry_run)

if __name__ == "__main__":
    main()
