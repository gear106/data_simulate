# UniSE + HCodec (UniTok) 适配说明

本说明记录将 UniSE 中的 BiCodec 替换为 UniTok 的 H-Codec 所需的关键改动。

## 1. 新增/修改文件

| 文件 | 说明 |
|------|------|
| `model/model_hcodec.py` | Lightning Module，替换原 `model.py` 中的 `Model`，使用 `HCodecTokenizer` 和 `LLM_SFT_HCodec`。 |
| `model/llm/llm_sft_hcodec.py` | LLM 核心，实现 delay pattern、4 层独立 embedding/output head。 |
| `conf/config_hcodec.yaml` | HCodec 版本配置文件。 |
| `train.py` | 已改为导入 `ModelHCodec` 并实例化 `ModelHCodec`。 |
| `test.py` | 已改为导入 `ModelHCodec` 并实例化 `ModelHCodec`。 |

原 `model.py` 和 `model/llm/llm_sft.py` 保持不动，方便双版本并存与回退。

## 2. Token 维度差异

BiCodec：
- `global_tokens`: `(B, 1, 32)` —— speaker 全局 token
- `semantic_tokens`: `(B, T)` —— 单层语义 token

H-Codec：
- `acoustic_codes`: `(B, 4, T_a)` —— 4 层 RVQ 声学 token
- `semantic_codes`: `(B, 4, T_s)` —— 4 层 RVQ 语义 token

默认时间分辨率：
- `mix_mel` 帧率 50 Hz（`hop_length=320`）。
- H-Codec 声学 token 帧率 25 Hz（encoder 总下采样 640 倍），即 `T_a = mix_mel.size(1) // 2`。
- H-Codec 语义 token 帧率 25 Hz（语义编码器在 HuBERT 50 Hz 特征上再下采样 2 倍），即 `T_s = mix_mel.size(1) // 2`。
- 若实际 H-Codec 配置不同，可在调用 `generate` 时显式传入 `acoustic_t` / `semantic_t`。

## 3. Delay Pattern

参考 UniTok / MusicGen (Copet et al., 2023) 论文：
> "the 4-layer acoustic and semantic tokens ... are first interleaved sequentially across time steps ... different shifts are applied across layers and special pad tokens occupy empty positions"

实现方式：

1. 先将 25 Hz 的声学帧和语义帧按时间步交错成 50 Hz 的统一序列 `Ec`：

```text
Ec = [A0, S0, A1, S1, A2, S2, ...]
```

2. 对 `Ec` 整体做 delay pattern（第 `q` 层向右偏移 `q` 位，空缺处用 `pad` 填充）。

3. 在 delay pattern 结果前拼一个 `codec_sos` 帧、后拼一个 `codec_eos` 帧（4 层都是同一个特殊 token），作为 MusicGen 式 pattern 的起止标记。

以 `n_q=4`、`T=4`（2 声学 + 2 语义帧）为例，每层序列形如：

```text
layer 0: [S, A0_0, S0_0, A1_0, S1_0,  p,    p,    p,    e]
layer 1: [S,  p,   A0_1, S0_1, A1_1, S1_1,  p,    p,    e]
layer 2: [S,  p,    p,   A0_2, S0_2, A1_2, S1_2,  p,    e]
layer 3: [S,  p,    p,    p,   A0_3, S0_3, A1_3, S1_3,  e]
```

按时间步交错展平后，序列开头为 `[S, S, S, S, ...]`，结尾为 `[..., e, e, e, e]`。

展平后长度：`L = (T + n_q + 1) * n_q`。

相关函数：
- `LLM_SFT_HCodec.build_interleaved_ec(acoustic_ids, semantic_ids)`：`(B, 4, T_a)` + `(B, 4, T_s)` → `(B, T, 4)`。
- `LLM_SFT_HCodec.split_interleaved_ec(Ec)`：`(B, T, 4)` → `(B, 4, T_a)` + `(B, 4, T_s)`。
- `LLM_SFT_HCodec.build_delay_pattern(codes)`：`(B, 4, T)` → `(B, L_core)` + mask，`L_core = (T + 3) * 4`。
- `LLM_SFT_HCodec.recover_codes_from_delay(delayed_ids, T)`：`(B, L_core)` → `(B, 4, T)`，用于推理后 detokenize。

## 4. 4 层独立 Embedding / Output Head

论文：
> "4 embedding layers handle 4-layer tokens respectively, and the embeddings of each layer are added up as the input of transformer layers. There are 4 output heads to predict the 4-layer logits of next time step."

实现：
- `codec_embeddings`: `ModuleList[nn.Embedding(vocab_size, hidden_size)]`，共 4 个。
- `output_heads`: `ModuleList[nn.Linear(hidden_size, vocab_size)]`，共 4 个。
- 序列位置 `p` 对应的 RVQ 层为 `p % 4`，据此选择 embedding 和 output head。

Vocab 定义（以 `codebook_size=1024` 为例）：
- `0 ~ 1023`: codebook token
- `1024`: pad token（delay pattern 空位）
- `1025`: codec_sos（序列起始帧，4 层共用）
- `1026`: codec_eos（序列结束帧，4 层共用）

训练序列：
- 先构造 `Ec = [A0, S0, A1, S1, ...]`
- 对 `Ec` 做 delay pattern 得到 `delayed_core`（长度 `L_core`）
- 前后拼接 `codec_sos` 帧和 `codec_eos` 帧，得到 `delayed_ids`（长度 `L = L_core + 2*n_q`）
- input: `delayed_ids[:, :-1]`
- target: `delayed_ids[:, 1:]`
- pad 位置在 loss 中被 mask。

## 5. 配置说明

使用 `conf/config_hcodec.yaml`。关键差异：

```yaml
# HCodec 权重文件路径，也可以是目录（自动补全 weights.pt）
codec_ckpt_dir: ./checkpoints_hcodec/weights.pt

llm_config:
  llm_base_config:
    # 删除原 BiCodec 的 global_size / semantic_size
    codebook_size: 1024
    hidden_size: 512
    ...
```

## 6. 训练与测试入口

`train.py` 和 `test.py` 已导入 `ModelHCodec`：

```python
from model.model_hcodec import ModelHCodec
model = ModelHCodec(config=config)
```

训练：
```bash
python train.py --config conf/config_hcodec.yaml
```

测试：
```bash
python test.py --config conf/config_hcodec.yaml --save_enhanced ./enhanced_hcodec
```

**注意：** 由于 special token、序列结构以及 `generate` 输出格式均已调整，旧的 BiCodec checkpoint 与当前 HCodec 实现不兼容，需要从头训练。

## 7. 注意事项

1. **HCodec 权重格式**
   - `HCodecTokenizer` 通过 `torch.load(pt_path)` 加载 Codec state_dict，请确保 `codec_ckpt_dir` 指向正确的 `.pt` 文件。

2. **时间分辨率**
   - 默认假设声学和语义 token 均为 25 Hz。若 H-Codec 配置不同（例如语义编码器不下采样），请在 `LLM_SFT_HCodec.generate()` 调用处传入 `acoustic_t` 和 `semantic_t`。对应 `mix_mel` 为 50 Hz 时，`T_a = T_s = mix_mel.size(1) // 2`。

3. **Delay pattern 与简单展平**
   - 当前实现严格按论文 delay pattern（层间 shift + pad）处理。
   - 若希望退回到简单展平方案，可替换 `build_delay_pattern` / `recover_codes_from_delay` 为直接的 transpose + reshape。

4. **特殊 token 预测**
   - `codec_sos` 和 `codec_eos` 是帧级特殊 token，在 delay pattern 前后各出现一帧（每层都是同一个 id），由对应位置的 output head 预测。
   - 推理时 `codec_sos` 帧和 `codec_eos` 帧（各 4 个位置）都被强制设为固定 id 并更新 KV-cache，避免采样误差。

5. **显存与序列长度**
   - Delay pattern 使核心序列长度从 `4*T` 增加到 `(T+3)*4`；再加上 sos/eos 帧，总长度为 `(T+5)*4`。
   - 5 秒音频、`mix_mel` 为 50 Hz 时，声学和语义 token 的 T 均为 125，交错后 `T = 250`，总长度约为 `(250+5)*4 = 1020`。
   - 请根据显存调整 `max_position_embeddings` 和 batch size。

## 8. 文件对应关系

| 原 BiCodec 流程 | HCodec 流程 |
|-----------------|-------------|
| `BiCodecTokenizer.tokenize` → `(global, semantic)` | `HCodecTokenizer.tokenize` → `(acoustic, semantic)` |
| `global_tokens.squeeze(1)` 作为 global_ids | 声学和语义帧交错成 `Ec`，再经 delay pattern 作为统一序列 |
| `semantic_tokens` 作为 semantic_ids | `Ec` 经 delay pattern 后由 LM 统一预测，再解交错回声学/语义 |
| `global_size=4096, semantic_size=8192` | `codebook_size=1024` |
| 1 个 shared embedding + 1 个 shared output head | 4 个 layer embedding + 4 个 layer output head |
| `tokenizer.detokenize(global_tokens, semantic_tokens)` | `tokenizer.detokenize(acoustic_codes, semantic_codes)` |
