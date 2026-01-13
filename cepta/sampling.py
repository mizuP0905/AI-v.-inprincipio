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
from typing import Optional, Tuple, Union

import torch


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(int(top_k), logits.size(-1))
        values, indices = torch.topk(logits, k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, indices, values)
        logits = masked

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(1, sorted_idx, sorted_logits)
        logits = masked

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    tokenizer: Optional[object] = None,
    generator: Optional[torch.Generator] = None,
    return_cache: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, object]]:
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_id", None)

    logits, cache = model.prefill(input_ids)
    next_token = sample_logits(
        logits[:, -1, :],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )

    if max_new_tokens <= 0:
        if return_cache:
            return input_ids, cache
        return input_ids

    B = input_ids.shape[0]
    alive = torch.ones(B, dtype=torch.bool, device=input_ids.device)
    generated = []

    for _ in range(max_new_tokens):
        token_in = next_token
        if eos_token_id is not None and not alive.all():
            eos_fill = torch.full_like(token_in, eos_token_id)
            token_in = torch.where(alive, token_in, eos_fill)

        logits, cache = model.decode_step(token_in, cache)
        generated.append(token_in)

        if eos_token_id is not None:
            alive = alive & (token_in != eos_token_id)
            if not alive.any():
                break

        next_token = sample_logits(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

    if generated:
        gen_tokens = torch.stack(generated, dim=1)
        output_ids = torch.cat([input_ids, gen_tokens], dim=1)
    else:
        output_ids = input_ids

    if return_cache:
        return output_ids, cache
    return output_ids


if __name__ == "__main__":
    class _DummyModel:
        def __init__(self) -> None:
            self.calls = 0

        def prefill(self, input_ids: torch.Tensor):
            B, T = input_ids.shape
            logits = torch.zeros(B, T, 8)
            return logits, {"pos_offset": T}

        def decode_step(self, token_ids: torch.Tensor, cache):
            self.calls += 1
            logits = torch.zeros(token_ids.shape[0], 1, 8)
            return logits, cache

    model = _DummyModel()
    ids = torch.randint(0, 8, (2, 3))
    out = generate(model, ids, max_new_tokens=2, temperature=0.0)
    assert out.shape[1] == 5
    print("Sampling sanity check passed.")
