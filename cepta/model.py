# Copyright (C) 2026 Kim Yoon Ki
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from .block import CeptaBlock
from .config import CeptaConfig, GraphConfig, SSMConfig
from .embedding import CeptaEmbedding, PositionalEncoding
from .sampling import generate as sampling_generate
from .ssm import LowRankSSM
from .synapse_graph import SynapseGraph


@dataclass
class CeptaCache:
    states: List[torch.Tensor]
    pos_offset: int = 0

    def to(self, device: torch.device) -> "CeptaCache":
        states = []
        for state in self.states:
            if state is None:
                states.append(None)
            else:
                states.append(state.to(device))
        return CeptaCache(states=states, pos_offset=self.pos_offset)


class CeptaLM(nn.Module):
    def __init__(self, config: CeptaConfig) -> None:
        super().__init__()
        config.finalize()
        self.config = config

        if config.graph_ssm.positions_mode != config.graph_mlp.positions_mode:
            raise ValueError("positions_mode must be consistent across graphs.")

        self.P = config.P
        self.alpha = config.alpha
        self.D = config.P * config.alpha

        self.embedding = CeptaEmbedding(
            vocab_size=config.vocab_size,
            P=config.P,
            alpha=config.alpha,
            D_out=self.D,
            use_ste=config.use_ste,
            ste_mode=config.ste_mode,
            ste_tau=config.ste_tau,
            dale_mode=config.dale_mode,
            e_ratio=config.e_ratio,
            neuron_type_seed=config.seed_base + 5,
        )
        if config.pos.mode == "A":
            self.positional = PositionalEncoding(self.D, config.pos.max_seq_len)
        else:
            self.positional = None

        positions = self._build_positions(
            num_layers=config.num_layers + 1,
            P=config.P,
            mode=config.graph_ssm.positions_mode,
            seed_base=config.seed_base,
        )
        self.register_buffer("positions", positions)

        self.blocks = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            graph_ssm = self._build_graph(
                layer_idx=layer_idx,
                role_id=1,
                graph_cfg=config.graph_ssm,
                cross_layer=True,
            )
            graph_mlp = self._build_graph(
                layer_idx=layer_idx,
                role_id=2,
                graph_cfg=config.graph_mlp,
                cross_layer=config.mlp_cross_layer,
            )
            ssm = self._build_ssm(config.ssm)
            neuron_type_seed = config.seed_base + 1000 * layer_idx + 30
            block = CeptaBlock(
                D=self.D,
                P=config.P,
                K=config.K,
                alpha=config.alpha,
                graph_ssm=graph_ssm,
                graph_mlp=graph_mlp,
                use_ste=config.use_ste,
                ste_mode=config.ste_mode,
                ste_tau=config.ste_tau,
                dale_mode=config.dale_mode,
                e_ratio=config.e_ratio,
                ssm=ssm,
                neuron_type_seed=neuron_type_seed,
            )
            self.blocks.append(block)

        self.lm_head = nn.Linear(self.D, config.vocab_size, bias=False)

    @staticmethod
    def _build_positions(
        num_layers: int, P: int, mode: str, seed_base: int
    ) -> torch.Tensor:
        positions = []
        for layer_idx in range(num_layers):
            seed = seed_base + 1000 * layer_idx + 90
            pos = CeptaLM._generate_positions(P, mode, seed)
            positions.append(pos)
        return torch.stack(positions, dim=0)

    @staticmethod
    def _generate_positions(P: int, mode: str, seed: int) -> torch.Tensor:
        if mode == "random":
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            return torch.rand(P, 2, generator=gen, dtype=torch.float32)
        if mode == "grid":
            side = int(torch.ceil(torch.sqrt(torch.tensor(float(P))))).item()
            coords = torch.linspace(0.0, 1.0, steps=side, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
            grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
            return grid[:P]
        raise ValueError(f"Unsupported positions mode: {mode}")

    def _build_graph(
        self,
        layer_idx: int,
        role_id: int,
        graph_cfg: GraphConfig,
        cross_layer: bool,
    ) -> SynapseGraph:
        if cross_layer:
            pos_src = self.positions[layer_idx]
            pos_tgt = self.positions[layer_idx + 1]
            same_layer = False
        else:
            pos_src = self.positions[layer_idx + 1]
            pos_tgt = self.positions[layer_idx + 1]
            same_layer = True

        seed = self.config.seed_base + 1000 * layer_idx + 10 * role_id
        src_idx = SynapseGraph.generate_src_idx(
            positions_src=pos_src,
            positions_tgt=pos_tgt,
            alpha_source=graph_cfg.alpha,
            K=graph_cfg.K,
            lambda_local=graph_cfg.lambda_local,
            rho_lr=graph_cfg.rho_lr,
            r_hub=graph_cfg.r_hub,
            rho_hub=graph_cfg.rho_hub,
            unique_per_target=graph_cfg.unique_per_target,
            allow_self=graph_cfg.allow_self,
            seed=seed,
            same_layer=same_layer,
        )
        return SynapseGraph(src_idx=src_idx, P_target=graph_cfg.P, K=graph_cfg.K)

    @staticmethod
    def _build_ssm(ssm_cfg: SSMConfig) -> LowRankSSM:
        return LowRankSSM(
            P=ssm_cfg.P,
            P_r=ssm_cfg.P_r,
            mode=ssm_cfg.mode,
            a_min=ssm_cfg.a_min,
            a_max=ssm_cfg.a_max,
            rms_norm=ssm_cfg.rms_norm,
            rms_eps=ssm_cfg.rms_eps,
            learnable_s0=ssm_cfg.learnable_s0,
        )

    def _prepare_cache(
        self,
        cache: Optional[CeptaCache],
        device: torch.device,
        batch_size: int,
    ) -> Tuple[List[torch.Tensor], int]:
        if cache is None:
            return [None] * len(self.blocks), 0

        if len(cache.states) != len(self.blocks):
            raise ValueError("cache.states length must match number of layers.")
        for state in cache.states:
            if state is None:
                continue
            if state.device != device:
                raise RuntimeError(
                    "Cache state is on a different device than inputs."
                )
            if state.shape[0] != batch_size:
                raise RuntimeError("Cache state batch size mismatch.")
        return cache.states, cache.pos_offset

    def _run_blocks(
        self,
        input_ids: torch.Tensor,
        states: List[torch.Tensor],
        pos_offset: int,
        return_local: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[dict]]:
        if return_local:
            x, _, embed_local = self.embedding(
                input_ids, return_ports=False, return_local=True
            )
        else:
            x = self.embedding(input_ids, return_ports=False)
            embed_local = None
        x = x.float()

        if self.positional is not None:
            x = self.positional(x, offset=pos_offset)

        if len(states) != len(self.blocks):
            raise ValueError("states length must match number of layers.")

        new_states = []
        block_locals = [] if return_local else None
        for idx, block in enumerate(self.blocks):
            if return_local:
                x, new_state, local = block(x, states[idx], return_local=True)
                block_locals.append(local)
            else:
                x, new_state = block(x, states[idx])
            new_states.append(new_state)

        logits = self.lm_head(x)
        if return_local:
            return logits, new_states, {"embed": embed_local, "blocks": block_locals}
        return logits, new_states, None

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[CeptaCache] = None,
        return_cache: bool = False,
        return_local: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, CeptaCache],
        Tuple[torch.Tensor, dict],
        Tuple[torch.Tensor, CeptaCache, dict],
    ]:
        if return_local:
            logits, new_cache, local = self.prefill(
                input_ids, cache=cache, return_local=True
            )
            if return_cache:
                return logits, new_cache, local
            return logits, local
        logits, new_cache = self.prefill(input_ids, cache=cache, return_local=False)
        if return_cache:
            return logits, new_cache
        return logits

    def prefill(
        self,
        input_ids: torch.Tensor,
        cache: Optional[CeptaCache] = None,
        return_local: bool = False,
    ) -> Union[Tuple[torch.Tensor, CeptaCache], Tuple[torch.Tensor, CeptaCache, dict]]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        states, pos_offset = self._prepare_cache(
            cache, input_ids.device, input_ids.shape[0]
        )
        logits, new_states, local = self._run_blocks(
            input_ids, states, pos_offset, return_local
        )
        new_cache = CeptaCache(new_states, pos_offset + input_ids.shape[1])
        if return_local:
            return logits, new_cache, local
        return logits, new_cache

    def decode_step(
        self,
        last_token_ids: torch.Tensor,
        cache: CeptaCache,
    ) -> Tuple[torch.Tensor, CeptaCache]:
        if cache is None:
            raise ValueError("cache is required for decode_step.")
        if last_token_ids.dim() == 1:
            last_token_ids = last_token_ids.unsqueeze(1)
        elif last_token_ids.dim() == 2 and last_token_ids.shape[1] == 1:
            pass
        else:
            raise ValueError("last_token_ids must have shape (B,) or (B, 1).")

        states, pos_offset = self._prepare_cache(
            cache, last_token_ids.device, last_token_ids.shape[0]
        )
        logits, new_states, _ = self._run_blocks(
            last_token_ids, states, pos_offset, return_local=False
        )
        new_cache = CeptaCache(new_states, pos_offset + 1)
        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        tokenizer: Optional[object] = None,
        generator: Optional[torch.Generator] = None,
        return_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, CeptaCache]]:
        return sampling_generate(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            tokenizer=tokenizer,
            generator=generator,
            return_cache=return_cache,
        )


if __name__ == "__main__":
    cfg = CeptaConfig(
        vocab_size=16,
        P=4,
        alpha=2,
        K=4,
        num_layers=2,
        seed_base=11,
    )
    model = CeptaLM(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (2, 5))
    logits, cache = model.prefill(prompt)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    step_logits, cache2 = model.decode_step(next_token, cache)
    full_logits = model.forward(torch.cat([prompt, next_token.unsqueeze(1)], dim=1))
    assert torch.allclose(step_logits[:, -1, :], full_logits[:, -1, :])
    assert cache2.pos_offset == prompt.shape[1] + 1
    print("Model prefill/decode sanity check passed.")
