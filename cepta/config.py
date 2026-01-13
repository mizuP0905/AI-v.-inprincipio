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
from typing import Optional


@dataclass
class GraphConfig:
    P: int
    alpha: int
    K: int = 64
    lambda_local: float = 1.0
    rho_lr: float = 0.1
    r_hub: float = 0.1
    rho_hub: float = 0.5
    unique_per_target: bool = True
    allow_self: bool = False
    positions_mode: str = "random"  # "random" or "grid"


@dataclass
class SSMConfig:
    P: int
    P_r: int
    mode: str = "mode1"  # "mode1" or "mode2"
    a_min: float = 0.0
    a_max: float = 1.0
    rms_norm: bool = False
    rms_eps: float = 1e-5
    learnable_s0: bool = False


@dataclass
class PositionalConfig:
    mode: str = "A"  # "A" (additive) or "B" (none)
    max_seq_len: int = 2048


@dataclass
class CeptaConfig:
    vocab_size: int
    P: int
    alpha: int
    K: int = 64
    num_layers: int = 4
    use_ste: bool = False
    ste_mode: str = "A"  # "A" or "B"
    ste_tau: float = 1.0
    dale_mode: bool = False
    e_ratio: float = 0.8
    seed_base: int = 1234
    mlp_cross_layer: bool = True
    z_ref: float = 1.0
    eps_z: float = 1e-6
    beta_r: float = 0.1
    beta_m: float = 0.1
    r_star: float = 0.1
    m_star: float = 0.1
    eta_sp: float = 0.01
    lambda_r: float = 1.0
    lambda_m: float = 1.0
    sp_min: float = -10.0
    sp_max: float = 10.0
    w_min: float = -1.0
    w_max: float = 1.0
    f_min: float = -1.0
    f_max: float = 1.0
    graph_ssm: Optional[GraphConfig] = None
    graph_mlp: Optional[GraphConfig] = None
    ssm: Optional[SSMConfig] = None
    pos: Optional[PositionalConfig] = None

    def build_graph_config(self) -> None:
        if self.graph_ssm is None:
            self.graph_ssm = GraphConfig(P=self.P, alpha=self.alpha, K=self.K)
        if self.graph_mlp is None:
            self.graph_mlp = GraphConfig(P=self.P, alpha=self.alpha, K=self.K)

    def build_ssm_config(self) -> None:
        if self.ssm is None:
            self.ssm = SSMConfig(P=self.P, P_r=max(1, self.P // 4))

    def build_pos_config(self) -> None:
        if self.pos is None:
            self.pos = PositionalConfig()

    def finalize(self) -> None:
        self.build_graph_config()
        self.build_ssm_config()
        self.build_pos_config()
