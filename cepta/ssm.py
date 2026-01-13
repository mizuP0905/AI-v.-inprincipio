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


class LowRankSSM(nn.Module):
    def __init__(
        self,
        P: int,
        P_r: int,
        mode: str = "mode1",
        a_min: float = 0.0,
        a_max: float = 1.0,
        rms_norm: bool = False,
        rms_eps: float = 1e-5,
        learnable_s0: bool = False,
    ) -> None:
        super().__init__()
        self.P = P
        self.P_r = P_r
        self.mode = mode
        self.a_min = a_min
        self.a_max = a_max
        self.rms_norm = rms_norm
        self.rms_eps = rms_eps
        self.learnable_s0 = learnable_s0

        self.V_r = nn.Parameter(torch.empty(P, P_r))
        self.W_lambda = nn.Parameter(torch.empty(P_r, P_r))
        self.b_lambda = nn.Parameter(torch.zeros(P_r))
        self.V_b = nn.Parameter(torch.empty(P, P_r))
        self.V_o = nn.Parameter(torch.empty(P_r, P))
        self.gamma = nn.Parameter(torch.zeros(P_r))

        if learnable_s0:
            self.s0 = nn.Parameter(torch.zeros(P_r))
        else:
            self.register_parameter("s0", None)

        nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.W_lambda)
        nn.init.xavier_uniform_(self.V_b)
        nn.init.xavier_uniform_(self.V_o)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.rms_eps).sqrt()
        return x_fp32 / rms

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.learnable_s0 and self.s0 is not None:
            if self.s0.device != device:
                raise RuntimeError(
                    "SSM s0 parameter is on a different device than inputs."
                )
            return self.s0.float().unsqueeze(0).expand(batch_size, -1)
        return torch.zeros(batch_size, self.P_r, device=device, dtype=torch.float32)

    def forward(
        self,
        t: torch.Tensor,
        f_gate: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # t, f_gate: (B, T, P)
        B, T, _ = t.shape
        t_fp32 = t.float()
        f_fp32 = f_gate.float()

        if state is None:
            s = self.init_state(B, t.device)
        else:
            s = state.float()

        outputs = []

        if self.mode == "mode2":
            decay = torch.sigmoid(self.gamma.float()).unsqueeze(0)

        for i in range(T):
            ti = t_fp32[:, i, :]
            fi = f_fp32[:, i, :]

            if self.rms_norm:
                ti = self._rms_norm(ti)

            if self.mode == "mode1":
                r = ti @ self.V_r.float()
                a = torch.sigmoid(r @ self.W_lambda.float() + self.b_lambda.float())
                a = a.clamp(min=self.a_min, max=self.a_max)
                s = a * s + (fi * ti) @ self.V_b.float()
            elif self.mode == "mode2":
                s = decay * s + (fi * ti) @ self.V_b.float()
            else:
                raise ValueError(f"Unsupported SSM mode: {self.mode}")

            outputs.append(s @ self.V_o.float())

        tilde_t = torch.stack(outputs, dim=1)
        return tilde_t, s
