URGENT2026 Track1 训练数据生成说明
=====================================

本目录包含一键生成模拟训练数据所需的脚本和配置文件。

1. 文件说明
-----------
generate_scp.py         - 根据原始音频目录自动生成 .scp 及配套文件
simulation_config.yaml  - simulation 参数配置文件（可自定义 SNR、RIR 概率等）
run_simulation.py       - 一键运行生成参数 + 合成音频的两步流程

2. 前置依赖
-----------
确保已安装以下 Python 包：
  pip install soundfile librosa numpy scipy torch tqdm pyyaml

3. 快速开始
-----------

步骤 A：生成 .scp 等输入文件
------------------------------
修改以下命令中的路径为你本机实际路径，然后运行：

  python gen_data/generate_scp.py \
      --speech_dir  "D:/code/data/VoiceBank/clean" \
      --noise_dir   "D:/code/data/DEMAND" \
      --rir_dir     "D:/code/data/RIR/rirs_noises/real_rirs_isotropic_noises" \
      --output_dir  "data/train_sources" \
      --speech_fs   16000 \
      --noise_fs    16000 \
      --rir_fs      16000

运行后会在 data/train_sources/ 下生成：
  - speech_sources.scp    干净语音列表
  - utt2spk               说话人映射
  - text                  文本转录（无则填 <not-available>）
  - noise_scoures.scp     噪声列表
  - wind_noise_scoures.scp 风噪列表（当前为空）
  - rirs.scp              房间脉冲响应列表
  - source_length.scp     音频帧数

步骤 B：生成模拟参数 (meta.tsv)
--------------------------------
  python simulation/generate_data_param.py --config gen_data/simulation_config.yaml

这会在 simulation_train/log/meta.tsv 中记录每条数据的：
  - 选用的 speech/noise/rir ID
  - SNR、augmentation 类型、采样率等

步骤 C：合成 clean / noisy 音频
--------------------------------
  python simulation/simulate_data_from_param.py \
      --config gen_data/simulation_config.yaml \
      --meta_tsv simulation_train/log/meta.tsv \
      --nj 4 \
      --chunksize 50 \
      --highpass True

合成后的音频保存在：
  - simulation_train/clean/   干净语音（可能已加 RIR）
  - simulation_train/noisy/   带噪语音

一键运行（推荐）
----------------
如果已经完成步骤 A，可直接用包装脚本一键执行 B+C：

  python gen_data/run_simulation.py --config gen_data/simulation_config.yaml --nj 4

4. 常见问题
-----------
Q: 提示 "No module named 'espnet2'"
A: 项目根目录已内置 mock 的 espnet2 模块。请确保在 urgent2026_challenge_track1-main
   根目录下运行命令，或保持 sys.path 包含项目根目录。

Q: 提示 "torchaudio.io not available"
A: 当前环境 torchaudio 版本缺少 io 模块。脚本会自动跳过 codec 增强并打印警告，
   不影响其他增强（RIR、噪声、clipping、packet_loss、bandwidth_limitation）的正常使用。

Q: 磁盘空间不足
A: repeat_per_utt 默认为 1，表示每条干净语音生成 1 对 clean/noisy。
   VoiceBank 约 800 条语音，合成后约占用 1~2 GB。若数据量大，请确保磁盘充足。

Q: 如何调整增强概率？
A: 编辑 gen_data/simulation_config.yaml：
   - prob_reverberation: 0.5   # RIR 混响概率
   - snr_low_bound / snr_high_bound  # SNR 范围 (dB)
   - augmentations 下的各 weight  # 各种失真类型的权重
   - num_augmentations  # 同时应用几种失真的概率分布
