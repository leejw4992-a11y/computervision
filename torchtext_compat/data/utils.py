"""torchtext.data.utils 호환 구현 (원본 0.18.0의 동작을 그대로 옮김)."""

import re
from functools import partial

__all__ = ["get_tokenizer"]


def _split_tokenizer(x):
    return x.split()


def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]


# 원본 torchtext/data/utils.py 의 정규식·치환쌍을 그대로 사용 (토큰화 결과 동일)
_patterns = [r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;", r"\:", r"\s+"]
_replacements = [" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "]
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))


def _basic_english_normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()


def get_tokenizer(tokenizer, language="en"):
    """torchtext.data.utils.get_tokenizer 와 동일한 토크나이저를 돌려줍니다.

    지원: None(공백 분리), 'basic_english', 'spacy', 그리고 임의의 callable.
    moses/toktok/revtok/subword 는 구현하지 않았습니다.
    """
    if tokenizer is None:
        return _split_tokenizer

    if tokenizer == "basic_english":
        if language != "en":
            raise ValueError("Basic normalization is only available for English(en)")
        return _basic_english_normalize

    if callable(tokenizer):
        return tokenizer

    if tokenizer == "spacy":
        import spacy

        try:
            nlp = spacy.load(language)
        except (IOError, OSError) as exc:
            raise OSError(
                f"spaCy 모델 '{language}' 를 찾을 수 없습니다. 설치: "
                f"python -m spacy download {language}"
            ) from exc
        return partial(_spacy_tokenize, spacy=nlp)

    raise ValueError(
        f"지원하지 않는 토크나이저입니다: {tokenizer!r} "
        "(None, 'basic_english', 'spacy', callable 만 구현되어 있습니다)"
    )
