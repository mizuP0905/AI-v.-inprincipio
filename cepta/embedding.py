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
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from .cepta_perceptron import CeptaPerceptronIndex
from .ops import flatten_ports


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.register_buffer("pos_cache", self._build_cache(max_len), persistent=True)

    def _build_cache(self, length: int) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.dim)
        )
        pe = torch.zeros(length, self.dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_length(self, needed: int) -> None:
        if needed <= self.pos_cache.shape[0]:
            return
        new_cache = self._build_cache(needed)
        self.pos_cache = new_cache.to(self.pos_cache.device)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, D = x.shape
        needed = offset + T
        self._ensure_length(needed)
        pos = self.pos_cache[offset:offset + T].unsqueeze(0).to(x.dtype)
        return x + pos


class CeptaEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        P: int,
        alpha: int,
        D_out: Optional[int] = None,
        use_ste: bool = False,
        ste_mode: str = "A",
        ste_tau: float = 1.0,
        dale_mode: bool = False,
        e_ratio: float = 0.8,
        neuron_type_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.P = P
        self.alpha = alpha
        self.D_emb = P * alpha
        self.D_out = self.D_emb if D_out is None else D_out
        self.perceptron = CeptaPerceptronIndex(
            P=P,
            alpha=alpha,
            vocab_size=vocab_size,
            use_ste=use_ste,
            ste_mode=ste_mode,
            ste_tau=ste_tau,
            dale_mode=dale_mode,
            e_ratio=e_ratio,
            neuron_type_seed=neuron_type_seed,
        )
        if self.D_out != self.D_emb:
            self.W_out = nn.Parameter(torch.empty(self.D_emb, self.D_out))
            nn.init.xavier_uniform_(self.W_out)
        else:
            self.register_parameter("W_out", None)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_ports: bool = False,
        return_local: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ]:
        u, f_hard, y = self.perceptron(input_ids)
        x_flat = flatten_ports(y)
        if self.W_out is not None:
            x_flat = x_flat @ self.W_out
        if return_ports or return_local:
            ports = y if return_ports else None
            local = (u, f_hard, y) if return_local else None
            return x_flat, ports, local
        return x_flat


if __name__ == "__main__":
    emb = CeptaEmbedding(vocab_size=16, P=4, alpha=2)
    input_ids = torch.randint(0, 16, (2, 5))
    x = emb(input_ids)
    x2, ports, local = emb(input_ids, return_ports=True, return_local=True)
    assert x.shape == (2, 5, 8)
    assert ports is not None and ports.shape == (2, 5, 4, 2)
    assert local is not None and local[0].shape == (2, 5, 4)
    loss = x2.sum()
    loss.backward()
    assert emb.perceptron.W_emb.grad is not None
    if torch.cuda.is_available():
        emb = emb.cuda()
        input_ids = input_ids.cuda()
        x = emb(input_ids)
        assert x.is_cuda
    print("Embedding sanity check passed.")
