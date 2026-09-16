"""Pylance 전용 `torchtext` 별칭 (런타임에는 import 되지 않습니다).

torchtext_compat.install_as_torchtext()는 sys.modules에 torchtext를 런타임에
등록하기 때문에 Pylance가 그것을 알 수 없어 `import torchtext`에 노란 줄이
그어집니다. 이 폴더(stubs/)는 .vscode/settings.json의
python.analysis.extraPaths에만 등록되어 있고 sys.path에는 들어가지 않으므로,
편집기 경고만 없애고 실제 실행 경로는 그대로 둡니다.
"""

from torchtext_compat import __version__ as __version__
from torchtext_compat import data as data
from torchtext_compat import datasets as datasets
from torchtext_compat import vocab as vocab
from torchtext_compat import Vocab as Vocab
from torchtext_compat import build_vocab_from_iterator as build_vocab_from_iterator
from torchtext_compat import get_tokenizer as get_tokenizer
from torchtext_compat import (
    disable_torchtext_deprecation_warning as disable_torchtext_deprecation_warning,
)
