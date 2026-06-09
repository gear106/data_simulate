# SCP 文件生成说明

本项目在在线合成模式下依赖三类 SCP 列表文件来定位训练/验证数据。本文档说明每种 SCP 的格式要求以及对应的离线生成方法。

---

## 1. SCP 格式说明

SCP 文件为纯文本，每行对应一条音频记录，列之间以空格分隔。

### 1.1 `speech.scp` — 干净语音列表（3 列）

| 列 | 含义 | 示例 |
|---|---|---|
| 1 | `utt_id`（唯一标识） | `emilia_0001` |
| 2 | `spk_id`（说话人标识） | `spk_1024` |
| 3 | `path`（音频路径） | `clean/emilia_0001.flac` |

**注意**：`path` 可以是绝对路径，也可以是与 `config.yaml` 中 `speech_scp_base_dir` 拼接的相对路径。

### 1.2 `noise.scp` — 噪声列表（5 列）

| 列 | 含义 | 示例 |
|---|---|---|
| 1 | `utt_id` | `fsd50k_237` |
| 2 | `fs`（采样率，Hz） | `16000` |
| 3 | `start`（起始采样点） | `0` |
| 4 | `frames`（总采样点数） | `320000` |
| 5 | `path`（音频路径） | `/data/noise/fsd50k/237.wav` |

**说明**：`start` 和 `frames` 用于支持从长噪声文件中随机切取片段；若整段使用，固定填 `0` 和总采样数即可。

### 1.3 `rir.scp` — 房间冲击响应列表（2 列）

| 列 | 含义 | 示例 |
|---|---|---|
| 1 | `utt_id` | `rir_001` |
| 2 | `path`（音频路径） | `/data/rir/simulated_rirs.wav` |

---

## 2. 生成脚本

使用 `generate_scp_offline.py` 离线生成上述三类 SCP 文件。

### 2.1 支持参数

```text
--speech_dir    Clean 音频目录（生成 speech.scp，3列）
--noise_dir     Noise 音频目录（生成 noise.scp，5列）
--rir_dir       RIR 音频目录（生成 rir.scp，2列）
--out_dir       SCP 文件输出目录（必填）
--speech_rel_to speech 路径以此为基准输出相对路径（对应 config.yaml 中 speech_scp_base_dir）
--num_workers   读取 noise 音频信息的并行 workers 数（默认 8）
```

### 2.2 使用示例

#### 示例 1：单独生成 speech.scp

```bash
python generate_scp_offline.py \
    --speech_dir /data/clean \
    --out_dir ./scp
```

输出：`./scp/speech.scp`

```text
emilia_0001 spk_1024 /data/clean/emilia_0001.flac
emilia_0002 spk_1024 /data/clean/emilia_0002.flac
...
```

#### 示例 2：单独生成 noise.scp

```bash
python generate_scp_offline.py \
    --noise_dir /data/noise \
    --out_dir ./scp
```

输出：`./scp/noise.scp`

```text
fsd50k_237 16000 0 320000 /data/noise/fsd50k/237.wav
dns5_noise_001 16000 0 480000 /data/noise/dns5/noise_001.wav
...
```

#### 示例 3：单独生成 rir.scp

```bash
python generate_scp_offline.py \
    --rir_dir /data/rir \
    --out_dir ./scp
```

输出：`./scp/rir.scp`

```text
rir_001 /data/rir/simulated_rirs/Room001.wav
rir_002 /data/rir/simulated_rirs/Room002.wav
...
```

#### 示例 4：同时生成三类，且 speech 输出相对路径

若 `config.yaml` 中配置了 `speech_scp_base_dir: /voxbox`，则 speech.scp 中建议保存相对路径：

```bash
python generate_scp_offline.py \
    --speech_dir /voxbox/clean \
    --noise_dir /data/noise \
    --rir_dir /data/rir \
    --out_dir ./scp \
    --speech_rel_to /voxbox
```

输出：`./scp/speech.scp`

```text
emilia_0001 spk_1024 clean/emilia_0001.flac
...
```

---

## 3. 与 config.yaml 的对应关系

在 `conf/config.yaml` 的 `dataset_config` 中引用生成的 SCP：

```yaml
dataset_config:
  train_kwargs:
    speech_scp_base_dir: /voxbox
    speech_scp_path:
      - /scp/speech.scp
    noise_scp_path:
      - /scp/noise.scp
    rir_scp_path: /scp/rir.scp
```

---

## 4. 注意事项

- `utt_id` 默认取音频文件名的 `stem`（不含扩展名）。若不同目录下存在同名文件，会产生重复 `utt_id` 警告，仅保留首次出现的文件。
- `noise.scp` 生成时需读取音频文件头获取采样率和总采样数，对大目录建议使用 `--num_workers` 加速。
- 所有路径均建议使用绝对路径，避免不同工作目录下解析异常。
