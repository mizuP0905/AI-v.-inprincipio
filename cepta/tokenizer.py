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
from typing import List, Optional, Sequence, Union

import torch

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - exercised in runtime.
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class DeepSeekV3Tokenizer:
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3-0324",
        trust_remote_code: bool = False,
    ) -> None:
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is required for DeepSeekV3Tokenizer. "
                "Install with: pip install transformers"
            ) from _TRANSFORMERS_IMPORT_ERROR

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, use_fast=True, trust_remote_code=trust_remote_code
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load tokenizer. "
                f"Install transformers and ensure the model is available: {model_name}"
            ) from exc

        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token is None:
                raise RuntimeError(
                    "Tokenizer missing eos token; cannot set pad token automatically."
                )
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.pad_id = int(self._tokenizer.pad_token_id)
        self.bos_id = (
            int(self._tokenizer.bos_token_id)
            if self._tokenizer.bos_token_id is not None
            else None
        )
        self.eos_id = (
            int(self._tokenizer.eos_token_id)
            if self._tokenizer.eos_token_id is not None
            else None
        )
        self.unk_id = (
            int(self._tokenizer.unk_token_id)
            if self._tokenizer.unk_token_id is not None
            else None
        )
        self.vocab_size = int(self._tokenizer.vocab_size)
        self._default_add_bos = bool(getattr(self._tokenizer, "add_bos_token", True))
        self._default_add_eos = bool(getattr(self._tokenizer, "add_eos_token", False))

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        add_bos = self._default_add_bos if add_bos is None else add_bos
        add_eos = self._default_add_eos if add_eos is None else add_eos

        bos_tokens = [self.bos_id] if add_bos and self.bos_id is not None else []
        eos_tokens = [self.eos_id] if add_eos and self.eos_id is not None else []
        extra = len(bos_tokens) + len(eos_tokens)

        if max_length is not None:
            max_text_length = max(max_length - extra, 0)
        else:
            max_text_length = None

        encoded = self._tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=truncation,
            max_length=max_text_length,
        )
        ids_list = encoded["input_ids"]
        combined = [bos_tokens + ids + eos_tokens for ids in ids_list]

        if max_length is None:
            pad_len = max((len(ids) for ids in combined), default=0)
        else:
            pad_len = max_length

        if padding or pad_len > 0:
            padded = []
            for ids in combined:
                if len(ids) < pad_len:
                    ids = ids + [self.pad_id] * (pad_len - len(ids))
                else:
                    ids = ids[:pad_len]
                padded.append(ids)
        else:
            padded = combined

        return torch.tensor(padded, dtype=torch.long)

    def decode(self, ids: torch.LongTensor) -> Union[str, List[str]]:
        if ids.dim() == 1:
            return self._tokenizer.decode(
                ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        return self._tokenizer.batch_decode(
            ids.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def apply_chat_template(
        self, messages: Sequence[dict], add_generation_prompt: bool = False
    ) -> str:
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support chat templates.")
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


if __name__ == "__main__":
    tokenizer = DeepSeekV3Tokenizer()
    ids = tokenizer.encode(
        ["Hello world", "Test sequence"],
        max_length=8,
        add_bos=True,
        add_eos=False,
    )
    text = tokenizer.decode(ids)
    assert isinstance(text, list) and len(text) == 2
    assert ids.shape == (2, 8)
    print("Tokenizer sanity check passed.")
