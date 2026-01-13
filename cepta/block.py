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
from typing import Optional, Tuple

import torch
from torch import nn

from .cepta_perceptron import CeptaPerceptronDense
from .ops import flatten_ports
from .rmsnorm import RMSNorm
from .ssm import LowRankSSM
from .synapse_graph import SynapseGraph


class CeptaBlock(nn.Module):
    def __init__(
        self,
        D: int,
        P: int,
        K: int,
        alpha: int,
        graph_ssm: SynapseGraph,
        graph_mlp: SynapseGraph,
        use_ste: bool,
        ste_mode: str,
        ste_tau: float,
        dale_mode: bool,
        e_ratio: float,
        ssm: LowRankSSM,
        neuron_type_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.P = P
        self.alpha = alpha
        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)
        self.graph_ssm = graph_ssm
        self.graph_mlp = graph_mlp
        self.perceptron_ssm = CeptaPerceptronDense(
            P=P,
            K=K,
            alpha=alpha,
            use_ste=use_ste,
            ste_mode=ste_mode,
            ste_tau=ste_tau,
            dale_mode=dale_mode,
            e_ratio=e_ratio,
            neuron_type_seed=neuron_type_seed,
        )
        self.perceptron_mlp = CeptaPerceptronDense(
            P=P,
            K=K,
            alpha=alpha,
            use_ste=use_ste,
            ste_mode=ste_mode,
            ste_tau=ste_tau,
            dale_mode=dale_mode,
            e_ratio=e_ratio,
            neuron_type_seed=None if neuron_type_seed is None else neuron_type_seed + 1,
        )
        self.ssm = ssm

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor],
        return_local: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.norm1(x)
        x_dend_ssm = self.graph_ssm(h1)
        u_ssm, f_ssm, _ = self.perceptron_ssm(x_dend_ssm)
        t_ssm = f_ssm * u_ssm
        tilde_t, new_state = self.ssm(t_ssm, f_ssm, state)

        f_out = self.perceptron_ssm.compute_f()
        y_ssm = (f_ssm * tilde_t).unsqueeze(-1) * f_out
        x_after = x + flatten_ports(y_ssm)

        h2 = self.norm2(x_after)
        x_dend_mlp = self.graph_mlp(h2)
        u_mlp, f_mlp, y_mlp = self.perceptron_mlp(x_dend_mlp)
        x_out = x_after + flatten_ports(y_mlp)

        if return_local:
            local = {
                "ssm": {"u": u_ssm, "f_hard": f_ssm, "x_dend": x_dend_ssm},
                "mlp": {"u": u_mlp, "f_hard": f_mlp, "x_dend": x_dend_mlp},
            }
            return x_out, new_state, local

        return x_out, new_state
