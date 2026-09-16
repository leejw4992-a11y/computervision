"""torchtext.vocab 호환 구현 (C++ Vocab 대신 dict/list 기반).

build_vocab_from_iterator 의 정렬 규칙(빈도 내림차순 → 사전순)과 specials 삽입
위치까지 원본과 동일하게 맞춰서, 같은 입력에 대해 같은 인덱스가 나옵니다.
"""

from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional

from torch import nn

__all__ = ["Vocab", "vocab", "build_vocab_from_iterator"]


class Vocab(nn.Module):
    """토큰 ↔ 인덱스 매핑. torchtext.vocab.Vocab 과 같은 인터페이스를 제공합니다.

    nn.Module 을 상속하므로 vocab(['a','b']) 호출과 torch.save(vocab) 이 그대로 됩니다.
    """

    def __init__(self, tokens: List[str], default_index: Optional[int] = None) -> None:
        super().__init__()
        self._itos: List[str] = list(tokens)
        self._stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self._itos)}
        if len(self._stoi) != len(self._itos):
            raise ValueError("중복된 토큰이 있습니다. vocab 은 유일한 토큰만 받습니다.")
        self._default_index = default_index

    # --- 조회 ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def __getitem__(self, token: str) -> int:
        idx = self._stoi.get(token)
        if idx is not None:
            return idx
        if self._default_index is None:
            raise RuntimeError(f"Token {token} not found and default index is not set")
        return self._default_index

    def forward(self, tokens: List[str]) -> List[int]:
        return self.lookup_indices(tokens)

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def lookup_token(self, index: int) -> str:
        if not 0 <= index < len(self._itos):
            raise RuntimeError(f"Specified index {index} is out of bounds of the size {len(self._itos)}")
        return self._itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.lookup_token(int(index)) for index in indices]

    def get_stoi(self) -> Dict[str, int]:
        return dict(self._stoi)

    def get_itos(self) -> List[str]:
        return list(self._itos)

    # --- 기본 인덱스 --------------------------------------------------------
    def set_default_index(self, index: Optional[int]) -> None:
        self._default_index = index

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    # --- 수정 ---------------------------------------------------------------
    def insert_token(self, token: str, index: int) -> None:
        if token in self._stoi:
            raise RuntimeError(f"Token {token} already exists in the Vocab with index: {self._stoi[token]}")
        if not 0 <= index <= len(self._itos):
            raise RuntimeError(f"Specified index {index} is out of bounds of the size {len(self._itos)}")
        self._itos.insert(index, token)
        self._stoi = {tok: i for i, tok in enumerate(self._itos)}

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self._itos))

    def __repr__(self) -> str:
        return f"Vocab(size={len(self._itos)}, default_index={self._default_index})"


def vocab(
    ordered_dict: Dict, min_freq: int = 1, specials: Optional[List[str]] = None, special_first: bool = True
) -> Vocab:
    """빈도 OrderedDict 로부터 Vocab 생성 (삽입 순서 유지, 원본과 동일한 규칙)."""
    specials = specials or []
    ordered_dict = OrderedDict(ordered_dict)
    for token in specials:
        ordered_dict.pop(token, None)

    tokens = [token for token, freq in ordered_dict.items() if freq >= min_freq]

    if special_first:
        tokens[0:0] = list(specials)
    else:
        tokens.extend(specials)

    return Vocab(tokens)


def build_vocab_from_iterator(
    iterator: Iterable,
    min_freq: int = 1,
    specials: Optional[List[str]] = None,
    special_first: bool = True,
    max_tokens: Optional[int] = None,
) -> Vocab:
    """토큰 리스트를 내놓는 iterator 로부터 Vocab 생성.

    정렬은 원본과 동일하게 (빈도 내림차순, 같으면 토큰 사전순) 입니다.
    """
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    specials = specials or []

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    if max_tokens is None:
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
    else:
        assert len(specials) < max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
        ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

    return vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
