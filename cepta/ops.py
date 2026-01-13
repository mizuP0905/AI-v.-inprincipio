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
import torch


def flatten_ports(y: torch.Tensor) -> torch.Tensor:
    # y: (B, T, P, alpha) -> (B, T, P*alpha)
    B, T, P, alpha = y.shape
    return y.reshape(B, T, P * alpha)


def unflatten_ports(x: torch.Tensor, P: int, alpha: int) -> torch.Tensor:
    # x: (B, T, P*alpha) -> (B, T, P, alpha)
    B, T, _ = x.shape
    return x.reshape(B, T, P, alpha)
