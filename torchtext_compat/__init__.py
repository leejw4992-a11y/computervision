"""torchtext 0.18 호환 레이어 (순수 파이썬, GPU 환경에서 사용)

왜 필요한가
-----------
torchtext 0.18.0은 컴파일된 확장(libtorchtext)이 torch 2.3의 C++ ABI에 링크되어
있어 torch 2.7(+cu128)에서는 import 자체가 실패합니다(WinError 127). 그런데
RTX 50 시리즈는 torch 2.7 이상에서만 동작하므로, "torchtext를 쓰면 CPU,
GPU를 쓰면 torchtext 없음"이라는 딜레마가 생깁니다.

이 모듈은 책/노트북에서 실제로 쓰이는 torchtext API만 순수 파이썬으로 다시
구현해서 그 딜레마를 없앱니다. 컴파일된 확장이 없으므로 torch 버전과 무관하게
동작하고, 따라서 메인 GPU 환경(.venv, torch 2.7+cu128)에서 그대로 쓸 수 있습니다.

사용법 1 — 책 코드를 한 글자도 안 고치고 쓰기 (권장)
---------------------------------------------------
    import torchtext_compat; torchtext_compat.install_as_torchtext()

    from torchtext.datasets import Multi30k          # 책 코드 그대로
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

install_as_torchtext()는 진짜 torchtext가 import 되는 환경(.venv-torchtext 커널)
에서는 아무 것도 하지 않습니다. 따라서 위 두 줄을 넣어두면 같은 노트북이
GPU 커널과 torchtext 커널 양쪽에서 모두 돌아갑니다.

사용법 2 — 명시적으로 쓰기
--------------------------
    from torchtext_compat.datasets import Multi30k
    from torchtext_compat.data.utils import get_tokenizer
    from torchtext_compat.vocab import build_vocab_from_iterator

구현 범위
---------
- data.utils.get_tokenizer  (None / 'basic_english' / 'spacy' / callable)
- data.functional.to_map_style_dataset
- vocab.Vocab, vocab.vocab, vocab.build_vocab_from_iterator
- datasets.Multi30k, datasets.AG_NEWS

GloVe/FastText 등 사전학습 벡터와 datapipe 관련 API는 구현하지 않았습니다.
"""

from torchtext_compat import data, datasets, vocab  # noqa: F401
from torchtext_compat.data.utils import get_tokenizer  # noqa: F401
from torchtext_compat.vocab import Vocab, build_vocab_from_iterator  # noqa: F401

__version__ = "0.18.0+compat"

_INSTALLED = False


def disable_torchtext_deprecation_warning():
    """진짜 torchtext와의 호환을 위한 no-op (이 모듈은 경고를 내지 않습니다)."""


def install_as_torchtext(force=False):
    """`import torchtext`가 이 호환 모듈을 가리키도록 sys.modules에 등록합니다.

    진짜 torchtext가 정상 import 되는 환경에서는 아무 것도 하지 않고 False를
    반환합니다(force=True면 그래도 덮어씁니다). 반환값은 "호환 모듈을 설치했는가".
    """
    global _INSTALLED
    import sys

    if not force:
        if "torchtext" in sys.modules and sys.modules["torchtext"] is not sys.modules[__name__]:
            return False
        try:
            import torchtext  # noqa: F401  진짜 torchtext가 살아있으면 그대로 둔다

            return False
        except Exception:
            pass

    self = sys.modules[__name__]
    sys.modules["torchtext"] = self
    sys.modules["torchtext.data"] = data
    sys.modules["torchtext.data.utils"] = data.utils
    sys.modules["torchtext.data.functional"] = data.functional
    sys.modules["torchtext.vocab"] = vocab
    sys.modules["torchtext.datasets"] = datasets
    _INSTALLED = True
    return True
