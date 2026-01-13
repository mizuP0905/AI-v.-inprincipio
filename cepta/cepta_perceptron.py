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
from torch.nn import functional as F


class CeptaPerceptronDense(nn.Module):
    def __init__(
        self,
        P: int,
        K: int,
        alpha: int,
        use_ste: bool = False,
        ste_mode: str = "A",
        ste_tau: float = 1.0,
        dale_mode: bool = False,
        e_ratio: float = 0.8,
        neuron_types: Optional[torch.Tensor] = None,
        neuron_type_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.P = P
        self.K = K
        self.alpha = alpha
        self.use_ste = use_ste
        self.ste_mode = ste_mode
        self.ste_tau = ste_tau
        self.dale_mode = dale_mode

        self.w = nn.Parameter(torch.empty(P, K))
        self.sp = nn.Parameter(torch.zeros(P))
        self.f_param = nn.Parameter(torch.empty(P, alpha))

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.f_param)

        self.register_buffer("r_ema", torch.zeros(P, dtype=torch.float32))
        self.register_buffer("m_ema", torch.zeros(P, dtype=torch.float32))

        if self.dale_mode:
            if neuron_types is None:
                gen = torch.Generator(device="cpu")
                if neuron_type_seed is not None:
                    gen.manual_seed(int(neuron_type_seed))
                neuron_types = (torch.rand(P, generator=gen) < e_ratio).long()
            neuron_types = neuron_types.view(P, 1).long()
            self.register_buffer("neuron_types", neuron_types)
        else:
            self.register_buffer("neuron_types", torch.empty(0), persistent=False)

    @staticmethod
    def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-6)
        return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

    @staticmethod
    def _row_mask(f_hard: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in ("all", "active", "inactive"):
            raise ValueError(f"Unsupported mask mode: {mode}")
        dims = tuple(range(f_hard.dim() - 1))
        mean_f = f_hard.float().mean(dim=dims)
        if mode == "all":
            mask = torch.ones_like(mean_f)
        elif mode == "active":
            mask = (mean_f > 0).float()
        else:
            mask = (mean_f <= 0).float()
        return mask.view(-1, 1)

    def compute_f(self) -> torch.Tensor:
        if not self.dale_mode:
            return self.f_param.float()
        if self.neuron_types.device != self.f_param.device:
            raise RuntimeError(
                "neuron_types buffer is on a different device than f_param. "
                "Move the module to the same device before calling forward."
            )
        types = self.neuron_types.to(dtype=torch.float32)
        sign = torch.where(types > 0, 1.0, -1.0)
        return F.softplus(self.f_param.float()) * sign

    def forward(
        self, x_dend: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x_dend: (B, T, P, K)
        x_fp32 = x_dend.float()
        w_fp32 = self.w.float()
        sp_fp32 = self.sp.float()

        u = (x_fp32 * w_fp32).sum(dim=-1)
        f_hard = (u >= sp_fp32).float()

        if self.use_ste:
            if self.ste_mode == "A":
                f_gate = f_hard + (u - u.detach())
            elif self.ste_mode == "B":
                mask = (u - sp_fp32).abs() < self.ste_tau
                f_gate = f_hard + (u - u.detach()) * mask.float()
            else:
                raise ValueError(f"Unsupported STE mode: {self.ste_mode}")
        else:
            f_gate = f_hard

        f_out = self.compute_f()
        y = (f_gate * u).unsqueeze(-1) * f_out
        return u, f_hard, y

    @torch.no_grad()
    def homeostatic_update(self, u: torch.Tensor, f_hard: torch.Tensor, cfg) -> None:
        if self.r_ema.device != self.sp.device:
            raise RuntimeError("Homeostasis buffers are on a different device than SP.")
        u_det = u.detach().float()
        f_det = f_hard.detach().float()
        sp_det = self.sp.detach().float()
        z_denom = float(cfg.z_ref) + float(cfg.eps_z)
        m_t = f_det * F.relu((u_det - sp_det) / z_denom)
        dims = tuple(range(f_det.dim() - 1))
        r_bar = f_det.mean(dim=dims)
        m_bar = m_t.mean(dim=dims)
        self.r_ema.mul_(1.0 - float(cfg.beta_r)).add_(float(cfg.beta_r) * r_bar)
        self.m_ema.mul_(1.0 - float(cfg.beta_m)).add_(float(cfg.beta_m) * m_bar)
        delta = float(cfg.lambda_r) * (self.r_ema - float(cfg.r_star))
        delta += float(cfg.lambda_m) * (self.m_ema - float(cfg.m_star))
        sp_fp32 = self.sp.data.float()
        sp_fp32.add_(float(cfg.eta_sp) * delta)
        sp_fp32.clamp_(float(cfg.sp_min), float(cfg.sp_max))
        self.sp.data.copy_(sp_fp32.to(self.sp.data.dtype))

    @torch.no_grad()
    def apply_grad_mask(self, f_hard: torch.Tensor, mask_w_mode: str, mask_f_mode: str) -> None:
        mask_w = self._row_mask(f_hard, mask_w_mode)
        mask_f = self._row_mask(f_hard, mask_f_mode)
        if self.w.grad is not None:
            self.w.grad.mul_(mask_w.to(self.w.grad.dtype))
        if self.f_param.grad is not None:
            self.f_param.grad.mul_(mask_f.to(self.f_param.grad.dtype))

    @torch.no_grad()
    def clip_parameters(self, cfg) -> None:
        self.w.data.clamp_(float(cfg.w_min), float(cfg.w_max))
        sp_fp32 = self.sp.data.float()
        sp_fp32.clamp_(float(cfg.sp_min), float(cfg.sp_max))
        self.sp.data.copy_(sp_fp32.to(self.sp.data.dtype))
        if not self.dale_mode:
            self.f_param.data.clamp_(float(cfg.f_min), float(cfg.f_max))
        else:
            max_mag = max(float(cfg.f_max), 0.0)
            mag = F.softplus(self.f_param.data.float())
            mag_clamped = mag.clamp(min=0.0, max=max_mag)
            f_new = self._softplus_inverse(mag_clamped)
            self.f_param.data.copy_(f_new.to(self.f_param.data.dtype))

    @torch.no_grad()
    def apply_f_l2_decay(self, lr_reg: float, f_hard: torch.Tensor, mask_mode: str) -> None:
        mask = self._row_mask(f_hard, mask_mode)
        f_fp32 = self.f_param.data.float()
        f_fp32.add_(-float(lr_reg) * f_fp32 * mask)
        self.f_param.data.copy_(f_fp32.to(self.f_param.data.dtype))

    def l2_stability_loss(self) -> torch.Tensor:
        # L2 stabilization for weights and thresholds.
        return 0.5 * (self.w.pow(2).mean() + self.sp.pow(2).mean())

    @torch.no_grad()
    def apply_oja_update(
        self,
        x_dend: torch.Tensor,
        activity: torch.Tensor,
        lr: float,
        mask_mode: str = "all",
        cfg=None,
    ) -> None:
        # Oja update: w <- w + lr * (y x - y^2 w), averaged over (B, T).
        x_fp32 = x_dend.float()
        y_fp32 = activity.float()
        xy = (y_fp32.unsqueeze(-1) * x_fp32).mean(dim=(0, 1))
        y2 = (y_fp32.pow(2).mean(dim=(0, 1))).unsqueeze(-1)
        delta = xy - y2 * self.w.data.float()
        mask = self._row_mask(activity, mask_mode)
        delta = delta * mask
        w_fp32 = self.w.data.float()
        w_fp32.add_(float(lr) * delta)
        self.w.data.copy_(w_fp32.to(self.w.data.dtype))
        if cfg is not None:
            self.clip_parameters(cfg)


class CeptaPerceptronIndex(nn.Module):
    def __init__(
        self,
        P: int,
        alpha: int,
        vocab_size: int,
        use_ste: bool = False,
        ste_mode: str = "A",
        ste_tau: float = 1.0,
        dale_mode: bool = False,
        e_ratio: float = 0.8,
        neuron_types: Optional[torch.Tensor] = None,
        neuron_type_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.P = P
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.use_ste = use_ste
        self.ste_mode = ste_mode
        self.ste_tau = ste_tau
        self.dale_mode = dale_mode

        self.W_emb = nn.Parameter(torch.empty(P, vocab_size))
        self.sp = nn.Parameter(torch.zeros(P))
        self.f_param = nn.Parameter(torch.empty(P, alpha))

        nn.init.xavier_uniform_(self.W_emb)
        nn.init.xavier_uniform_(self.f_param)

        self.register_buffer("r_ema", torch.zeros(P, dtype=torch.float32))
        self.register_buffer("m_ema", torch.zeros(P, dtype=torch.float32))

        if self.dale_mode:
            if neuron_types is None:
                gen = torch.Generator(device="cpu")
                if neuron_type_seed is not None:
                    gen.manual_seed(int(neuron_type_seed))
                neuron_types = (torch.rand(P, generator=gen) < e_ratio).long()
            neuron_types = neuron_types.view(P, 1).long()
            self.register_buffer("neuron_types", neuron_types)
        else:
            self.register_buffer("neuron_types", torch.empty(0), persistent=False)

    @staticmethod
    def _softplus_inverse(x: torch.Tensor) -> torch.Tensor:
        x = x.clamp_min(1e-6)
        return torch.where(x > 20.0, x, torch.log(torch.expm1(x)))

    @staticmethod
    def _row_mask(f_hard: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in ("all", "active", "inactive"):
            raise ValueError(f"Unsupported mask mode: {mode}")
        dims = tuple(range(f_hard.dim() - 1))
        mean_f = f_hard.float().mean(dim=dims)
        if mode == "all":
            mask = torch.ones_like(mean_f)
        elif mode == "active":
            mask = (mean_f > 0).float()
        else:
            mask = (mean_f <= 0).float()
        return mask.view(-1, 1)

    def compute_f(self) -> torch.Tensor:
        if not self.dale_mode:
            return self.f_param.float()
        if self.neuron_types.device != self.f_param.device:
            raise RuntimeError(
                "neuron_types buffer is on a different device than f_param. "
                "Move the module to the same device before calling forward."
            )
        types = self.neuron_types.to(dtype=torch.float32)
        sign = torch.where(types > 0, 1.0, -1.0)
        return F.softplus(self.f_param.float()) * sign

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must be torch.long for index embedding.")
        u = self.W_emb.t()[input_ids]
        u_fp32 = u.float()
        sp_fp32 = self.sp.float()
        f_hard = (u_fp32 >= sp_fp32).float()

        if self.use_ste:
            if self.ste_mode == "A":
                f_gate = f_hard + (u_fp32 - u_fp32.detach())
            elif self.ste_mode == "B":
                mask = (u_fp32 - sp_fp32).abs() < self.ste_tau
                f_gate = f_hard + (u_fp32 - u_fp32.detach()) * mask.float()
            else:
                raise ValueError(f"Unsupported STE mode: {self.ste_mode}")
        else:
            f_gate = f_hard

        f_out = self.compute_f()
        y = (f_gate * u_fp32).unsqueeze(-1) * f_out
        return u_fp32, f_hard, y

    @torch.no_grad()
    def homeostatic_update(self, u: torch.Tensor, f_hard: torch.Tensor, cfg) -> None:
        if self.r_ema.device != self.sp.device:
            raise RuntimeError("Homeostasis buffers are on a different device than SP.")
        u_det = u.detach().float()
        f_det = f_hard.detach().float()
        sp_det = self.sp.detach().float()
        z_denom = float(cfg.z_ref) + float(cfg.eps_z)
        m_t = f_det * F.relu((u_det - sp_det) / z_denom)
        dims = tuple(range(f_det.dim() - 1))
        r_bar = f_det.mean(dim=dims)
        m_bar = m_t.mean(dim=dims)
        self.r_ema.mul_(1.0 - float(cfg.beta_r)).add_(float(cfg.beta_r) * r_bar)
        self.m_ema.mul_(1.0 - float(cfg.beta_m)).add_(float(cfg.beta_m) * m_bar)
        delta = float(cfg.lambda_r) * (self.r_ema - float(cfg.r_star))
        delta += float(cfg.lambda_m) * (self.m_ema - float(cfg.m_star))
        sp_fp32 = self.sp.data.float()
        sp_fp32.add_(float(cfg.eta_sp) * delta)
        sp_fp32.clamp_(float(cfg.sp_min), float(cfg.sp_max))
        self.sp.data.copy_(sp_fp32.to(self.sp.data.dtype))

    @torch.no_grad()
    def apply_grad_mask(self, f_hard: torch.Tensor, mask_w_mode: str, mask_f_mode: str) -> None:
        mask_w = self._row_mask(f_hard, mask_w_mode)
        mask_f = self._row_mask(f_hard, mask_f_mode)
        if self.W_emb.grad is not None:
            self.W_emb.grad.mul_(mask_w.to(self.W_emb.grad.dtype))
        if self.f_param.grad is not None:
            self.f_param.grad.mul_(mask_f.to(self.f_param.grad.dtype))

    @torch.no_grad()
    def clip_parameters(self, cfg) -> None:
        self.W_emb.data.clamp_(float(cfg.w_min), float(cfg.w_max))
        sp_fp32 = self.sp.data.float()
        sp_fp32.clamp_(float(cfg.sp_min), float(cfg.sp_max))
        self.sp.data.copy_(sp_fp32.to(self.sp.data.dtype))
        if not self.dale_mode:
            self.f_param.data.clamp_(float(cfg.f_min), float(cfg.f_max))
        else:
            max_mag = max(float(cfg.f_max), 0.0)
            mag = F.softplus(self.f_param.data.float())
            mag_clamped = mag.clamp(min=0.0, max=max_mag)
            f_new = self._softplus_inverse(mag_clamped)
            self.f_param.data.copy_(f_new.to(self.f_param.data.dtype))

    @torch.no_grad()
    def apply_f_l2_decay(self, lr_reg: float, f_hard: torch.Tensor, mask_mode: str) -> None:
        mask = self._row_mask(f_hard, mask_mode)
        f_fp32 = self.f_param.data.float()
        f_fp32.add_(-float(lr_reg) * f_fp32 * mask)
        self.f_param.data.copy_(f_fp32.to(self.f_param.data.dtype))


if __name__ == "__main__":
    class _Cfg:
        z_ref = 1.0
        eps_z = 1e-6
        beta_r = 1.0
        beta_m = 1.0
        r_star = 0.1
        m_star = 0.1
        eta_sp = 0.5
        lambda_r = 1.0
        lambda_m = 0.0
        sp_min = -10.0
        sp_max = 10.0
        w_min = -1.0
        w_max = 1.0
        f_min = -1.0
        f_max = 1.0

    cfg = _Cfg()
    perceptron = CeptaPerceptronDense(P=4, K=3, alpha=2)
    u = torch.ones(2, 5, 4) * 2.0
    f = torch.ones(2, 5, 4)
    sp_before = perceptron.sp.detach().clone()
    perceptron.homeostatic_update(u, f, cfg)
    sp_after = perceptron.sp.detach()
    assert torch.all(sp_after >= sp_before)
    print("Perceptron homeostasis sanity check passed.")
