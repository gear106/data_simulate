import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

from .llm import CustomLlamaModel


class LLM_SFT_HCodec(CustomLlamaModel):
    """SFT LLM for HCodec tokens with interleaved delay pattern.

    Following the UniTok paper:
      - HCodec produces 4-layer acoustic and semantic tokens (B, 4, T).
      - Acoustic and semantic frames are interleaved into a single 50 Hz Ec
        sequence: [codec_sos, A0, S0, A1, S1, ..., codec_eos].
      - A single delay pattern is applied to Ec, with special pad tokens
        occupying empty positions.
      - The LM backbone uses 4 independent embedding layers and 4 independent
        output heads, one per RVQ layer.
      - Embeddings for each layer are selected by position.
    """

    NUM_QUANTIZERS = 4
    NUM_SPECIAL_TOKENS = 3  # pad, codec_sos, codec_eos

    def __init__(
        self,
        num_tasks: int = 1,
        task_map: dict = {
            'se': 0,
        },
        feats_dim: int = 768,
        llm_base_config: dict = {},
    ):
        # HCodec uses a single shared codebook size for all RVQ layers.
        self.codebook_size = llm_base_config.get('codebook_size', 1024)
        # Dummy values to satisfy parent __init__ signature; we override them.
        hcodec_config = llm_base_config.copy()
        hcodec_config['global_size'] = self.codebook_size
        hcodec_config['semantic_size'] = self.codebook_size
        super().__init__(**hcodec_config)

        # Override vocab size and special token ids.
        self.vocab_size = self.codebook_size + self.NUM_SPECIAL_TOKENS
        self.config.vocab_size = self.vocab_size
        self.pad_token_id = self.codebook_size
        self.codec_sos_token_id = self.codebook_size + 1
        self.codec_eos_token_id = self.codebook_size + 2

        # 4 independent embedding layers, one per RVQ layer.
        del self.codec_embedding
        self.codec_embeddings = nn.ModuleList([
            nn.Embedding(self.vocab_size, self.config.hidden_size)
            for _ in range(self.NUM_QUANTIZERS)
        ])

        # 4 independent output heads, one per RVQ layer.
        del self.output_head
        self.output_heads = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.vocab_size, bias=False)
            for _ in range(self.NUM_QUANTIZERS)
        ])

        self.task_map = task_map

        # Condition / task embeddings.
        self.task_embedding = nn.Embedding(num_tasks, self.config.hidden_size)
        self.enroll_sos_embedding = nn.Embedding(1, self.config.hidden_size)
        self.adapter = nn.Linear(feats_dim, self.config.hidden_size)

    # ------------------------------------------------------------------
    # Delay pattern helpers
    # ------------------------------------------------------------------
    def build_delay_pattern(
        self,
        codes: Union[torch.IntTensor, torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert (B, 4, T) RVQ codes into a flattened delay pattern.

        Returns:
            delayed_ids: (B, (T + 3) * 4) flattened token ids with pad tokens.
            mask:        (B, (T + 3) * 4) bool mask, True for non-pad positions.
        """
        b, n_q, t = codes.shape
        assert n_q == self.NUM_QUANTIZERS
        delayed = torch.full(
            (b, n_q, t + n_q - 1),
            self.pad_token_id,
            dtype=codes.dtype,
            device=codes.device,
        )
        for q in range(n_q):
            delayed[:, q, q:q + t] = codes[:, q, :]
        # Interleave across time steps: (B, T+3, 4) -> (B, (T+3)*4)
        delayed_flat = delayed.transpose(1, 2).reshape(b, -1)
        mask = delayed_flat != self.pad_token_id
        return delayed_flat, mask

    def recover_codes_from_delay(
        self,
        delayed_ids: Union[torch.IntTensor, torch.LongTensor],
        t: int,
    ) -> torch.Tensor:
        """Recover (B, 4, T) RVQ codes from a flattened delay pattern."""
        b = delayed_ids.size(0)
        n_q = self.NUM_QUANTIZERS
        delayed_2d = delayed_ids.reshape(b, t + n_q - 1, n_q)
        codes = torch.zeros(b, n_q, t, dtype=delayed_ids.dtype, device=delayed_ids.device)
        for q in range(n_q):
            codes[:, q, :] = delayed_2d[:, q:q + t, q]
        return codes

    def build_interleaved_ec(
        self,
        acoustic_ids: Union[torch.IntTensor, torch.LongTensor],
        semantic_ids: Union[torch.IntTensor, torch.LongTensor],
    ) -> torch.Tensor:
        """Build interleaved 50 Hz Ec from 25 Hz acoustic/semantic tokens.

        Ec = [A0, S0, A1, S1, A2, S2, ...]
        """
        b, n_q, t_a = acoustic_ids.shape
        _, _, t_s = semantic_ids.shape
        assert n_q == self.NUM_QUANTIZERS
        if t_a != t_s:
            raise ValueError(
                f"acoustic and semantic time steps must match: got {t_a} and {t_s}"
            )
        t = t_a + t_s  # interleaved acoustic/semantic frames
        Ec = torch.empty(
            (b, t, n_q),
            dtype=acoustic_ids.dtype,
            device=acoustic_ids.device,
        )
        # Even positions are acoustic; odd positions are semantic.
        Ec[:, ::2, :] = acoustic_ids.transpose(1, 2)
        Ec[:, 1::2, :] = semantic_ids.transpose(1, 2)
        return Ec

    def split_interleaved_ec(
        self,
        Ec: Union[torch.IntTensor, torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split interleaved Ec back into acoustic and semantic codes.

        Returns:
            acoustic_ids: (B, 4, T_a)
            semantic_ids: (B, 4, T_s)
        """
        t = Ec.size(1)
        if t % 2 != 0:
            raise ValueError(f"Ec time steps {t} must be even")
        t_a = t // 2
        acoustic_ids = Ec[:, ::2, :].transpose(1, 2)
        semantic_ids = Ec[:, 1::2, :].transpose(1, 2)
        return acoustic_ids, semantic_ids

    def _layer_ids_for_length(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the RVQ layer index for each position in a length-L sequence."""
        return torch.arange(length, device=device) % self.NUM_QUANTIZERS

    def _layer_ids_for_positions(self, start_pos: int, length: int, device: torch.device) -> torch.Tensor:
        """Return the RVQ layer index for positions [start_pos, start_pos + length)."""
        return (torch.arange(start_pos, start_pos + length, device=device) % self.NUM_QUANTIZERS)

    def embed_with_delay(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Select embeddings from the 4 layer-specific lookup tables by position."""
        b, length = input_ids.shape
        layer_ids = self._layer_ids_for_positions(start_pos, length, input_ids.device)
        embeds = torch.zeros(b, length, self.config.hidden_size, device=input_ids.device)
        for q in range(self.NUM_QUANTIZERS):
            mask = layer_ids == q
            if mask.any():
                embeds[:, mask, :] = self.codec_embeddings[q](input_ids[:, mask])
        return embeds

    def output_with_delay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the 4 layer-specific output heads by position."""
        b, length, _ = hidden_states.shape
        layer_ids = self._layer_ids_for_length(length, hidden_states.device)
        logits = torch.zeros(b, length, self.vocab_size, device=hidden_states.device)
        for q in range(self.NUM_QUANTIZERS):
            mask = layer_ids == q
            if mask.any():
                logits[:, mask, :] = self.output_heads[q](hidden_states[:, mask, :])
        return logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def loss_function(self, logits, target, mask):
        logits = logits.float()
        b, length, size = logits.shape
        logits_flat = logits.reshape(-1, size)
        target_flat = target.reshape(-1)
        mask_flat = mask.reshape(-1)

        if not mask_flat.any():
            return logits.sum() * 0.0

        confidence = 1.0 - self.label_smoothing
        with torch.no_grad():
            true_dist = torch.full_like(logits_flat, self.label_smoothing / (size - 1))
            true_dist.scatter_(1, target_flat.unsqueeze(1), confidence)

        loss = F.kl_div(
            F.log_softmax(logits_flat[mask_flat], dim=-1),
            true_dist[mask_flat],
            reduction='batchmean',
        )
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        task_name: str,
        enroll_mel: torch.Tensor,
        enroll_feats: torch.Tensor,
        mix_mel: torch.Tensor,
        mix_feats: torch.Tensor,
        acoustic_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, 4, T_a)
        semantic_ids: Union[torch.IntTensor, torch.LongTensor],  # (B, 4, T_s)
    ):
        # Interleave acoustic/semantic frames into a single 50 Hz Ec, apply the
        # delay pattern, then prepend a codec_sos frame and append a codec_eos
        # frame (one per RVQ layer), matching MusicGen's pattern layout.
        Ec = self.build_interleaved_ec(acoustic_ids, semantic_ids)
        delayed_ids, _ = self.build_delay_pattern(Ec.transpose(1, 2))

        b = delayed_ids.size(0)
        device = delayed_ids.device
        n_q = self.NUM_QUANTIZERS
        t_ec = Ec.size(1)
        seq_steps = t_ec + n_q - 1  # steps from the core delay pattern

        # Reshape to (B, seq_steps, n_q) to add sos/eos frames.
        delayed_2d = delayed_ids.reshape(b, seq_steps, n_q)
        delayed_with_se = torch.full(
            (b, seq_steps + 2, n_q),
            self.pad_token_id,
            dtype=delayed_ids.dtype,
            device=device,
        )
        delayed_with_se[:, 0, :] = self.codec_sos_token_id
        delayed_with_se[:, 1:-1, :] = delayed_2d
        delayed_with_se[:, -1, :] = self.codec_eos_token_id
        delayed_ids = delayed_with_se.reshape(b, -1)
        mask = delayed_ids != self.pad_token_id

        # Causal LM: input is delayed sequence without last token; target is
        # delayed sequence shifted by one position.
        input_ids = delayed_ids[:, :-1]
        target_ids = delayed_ids[:, 1:]
        loss_mask = mask[:, 1:]

        # Condition embeddings.
        task_embeds = self.task_embedding(
            torch.full((mix_mel.size(0), 1), self.task_map[task_name], dtype=torch.int64, device=device)
        )
        mixture = self.adapter(mix_feats)
        mix_sos_embeds = self.mix_sos_embedding(
            torch.full((mixture.size(0), 1), 0, dtype=torch.int64, device=device)
        )

        if enroll_mel is not None:
            enroll = self.adapter(enroll_feats)
            enroll_sos_embeds = self.enroll_sos_embedding(
                torch.full((enroll.size(0), 1), 0, dtype=torch.int64, device=device)
            )
            cond_embeds = torch.cat(
                [task_embeds, enroll_sos_embeds, enroll, mix_sos_embeds, mixture], dim=1
            )
        else:
            cond_embeds = torch.cat([task_embeds, mix_sos_embeds, mixture], dim=1)

        inputs_embeds = torch.cat([cond_embeds, self.embed_with_delay(input_ids)], dim=1)

        outputs = self.llm_forward(inputs_embeds)
        hidden_states = outputs.last_hidden_state[:, -target_ids.size(-1):, :]

        logits = self.output_with_delay(hidden_states)

        loss = self.loss_function(logits, target_ids, loss_mask)
        acc = ((logits.argmax(-1) == target_ids) & loss_mask).float().mean()

        return loss, acc

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        task_name: str,
        enroll_mel: torch.Tensor,
        enroll_feats: torch.Tensor,
        mix_mel: torch.Tensor,
        mix_feats: torch.Tensor,
        acoustic_t: int = None,
        semantic_t: int = None,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ):
        """Generate flattened delay-pattern tokens for the interleaved Ec sequence.

        Defaults assume mix_mel is 50 Hz and both HCodec acoustic and semantic
        tokens are 25 Hz. Pass ``acoustic_t`` / ``semantic_t`` to override.
        """
        if acoustic_t is None:
            acoustic_t = mix_mel.size(1) // 2
        if semantic_t is None:
            semantic_t = mix_mel.size(1) // 2

        n_q = self.NUM_QUANTIZERS
        t_ec = acoustic_t + semantic_t  # interleaved acoustic/semantic frames
        total_len = (t_ec + n_q + 1) * n_q  # + sos frame + eos frame

        device = mix_mel.device
        b = mix_mel.size(0)

        # Condition embeddings.
        task_embeds = self.task_embedding(
            torch.full((b, 1), self.task_map[task_name], dtype=torch.int64, device=device)
        )
        mixture = self.adapter(mix_feats)
        mix_sos_embeds = self.mix_sos_embedding(
            torch.full((b, 1), 0, dtype=torch.int64, device=device)
        )

        if enroll_mel is not None:
            enroll = self.adapter(enroll_feats)
            enroll_sos_embeds = self.enroll_sos_embedding(
                torch.full((b, 1), 0, dtype=torch.int64, device=device)
            )
            cond_embeds = torch.cat(
                [task_embeds, enroll_sos_embeds, enroll, mix_sos_embeds, mixture], dim=1
            )
        else:
            cond_embeds = torch.cat([task_embeds, mix_sos_embeds, mixture], dim=1)

        current_output = self.llm_forward(cond_embeds, past_key_values=None, use_cache=True)
        past_key_values = current_output.past_key_values

        # Relative position within the generated token sequence. The first
        # generated token (codec_sos) is at relative position 0; subsequent
        # tokens increment by 1. Embedding/output head are selected by this
        # relative position so that inference matches training.
        rel_pos = 0

        # Helper to run one autoregressive step at a known relative position.
        def step(input_ids: torch.Tensor, pos: int) -> torch.Tensor:
            nonlocal past_key_values
            inputs_embeds = self.embed_with_delay(input_ids, start_pos=pos)
            current_output = self.llm_forward(
                inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = current_output.past_key_values
            hidden = current_output.last_hidden_state[:, -1:, :]  # (B, 1, H)
            q = int(pos % n_q)
            logits = self.output_heads[q](hidden)  # (B, 1, V)
            next_token = self.sample_logits(
                logits.squeeze(1),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
            )
            return next_token

        # The delayed sequence starts with a full codec_sos frame and ends with a
        # full codec_eos frame; both are emitted deterministically.
        input_ids = torch.full(
            (b, 1), self.codec_sos_token_id, dtype=torch.long, device=device
        )
        generated = []
        for pos in range(total_len - 1):
            next_token = step(input_ids, rel_pos)
            # Force the first full frame (positions 1..n_q-1) to codec_sos.
            if pos < n_q - 1:
                next_token = torch.full(
                    (b, 1), self.codec_sos_token_id, dtype=torch.long, device=device
                )
            # Force the last full frame (positions total_len-n_q..total_len-1) to codec_eos.
            if pos >= total_len - n_q - 1:
                next_token = torch.full(
                    (b, 1), self.codec_eos_token_id, dtype=torch.long, device=device
                )
            generated.append(next_token)
            input_ids = next_token
            rel_pos += 1
        delayed_ids = torch.cat(
            [torch.full((b, 1), self.codec_sos_token_id, dtype=torch.long, device=device)]
            + generated,
            dim=-1,
        )

        # Recover interleaved Ec: strip sos/eos frames, undo delay pattern, split.
        delayed_2d = delayed_ids.reshape(b, t_ec + n_q + 1, n_q)
        delayed_core = delayed_2d[:, 1:-1, :]  # remove sos and eos frames
        delayed_core_flat = delayed_core.reshape(b, -1)
        Ec = self.recover_codes_from_delay(delayed_core_flat, t_ec).transpose(1, 2)
        acoustic_ids, semantic_ids = self.split_interleaved_ec(Ec)
        return acoustic_ids, semantic_ids
