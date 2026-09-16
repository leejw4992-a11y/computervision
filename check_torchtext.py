"""torchtext 환경 점검 스크립트.

두 환경 어디서 돌려도 됩니다.
  - 메인 GPU 환경 (.venv, torch 2.7+cu128) -> torchtext_compat 호환 모듈을 검사
  - torchtext 환경 (.venv-torchtext, torch 2.3 CPU) -> 진짜 torchtext 0.18 을 검사

터미널에서:
    .venv\\Scripts\\python.exe check_torchtext.py
    .venv-torchtext\\Scripts\\python.exe check_torchtext.py

노트북 셀에서:
    from check_torchtext import run_checks
    run_checks()

기대값(vocab 크기, 토큰 목록, 데이터 해시)은 진짜 torchtext 0.18.0 의 실제 출력에서
뽑은 것이라, 호환 모듈이 원본과 다르게 동작하면 바로 FAIL 로 드러납니다.
"""

import hashlib
import json
import sys
import time

DATA_ROOT = "datasets/torchtext"

# --- 진짜 torchtext 0.18.0 이 내놓은 값 (이 스크립트의 정답지) -------------------
EXPECTED = {
    "multi30k_counts": [29000, 1014],
    "multi30k_hash": "7d31b1067e930483",
    "tok_basic": ["you", "can", "'", "t", "install", "torchtext", "using", "pip", "(", "v2", ".", "0", ")", "!"],
    "tok_de_len": 13,
    "tok_en": ["Two", "young", ",", "White", "males", "are", "outside", "near", "many", "bushes", "."],
    "vocab_de_len": 8014,
    "vocab_de_hash": "fc378abefe3f2cda",
    "vocab_en_len": 6191,
    "vocab_en_hash": "320490d8716cf5bb",
    "vocab_head": ["<unk>", "<pad>", "<bos>", "<eos>"],
    "ag_news_len": 120000,
    "ag_news_first": 3,
}

_results = []


def _out(text=""):
    """cp949 콘솔에서도 죽지 않게 출력 (독일어 움라우트 등)."""
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc) + "\n")


def _check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    mark = "PASS" if ok else "FAIL"
    _out("  [" + mark + "] " + name + ("  " + detail if detail else ""))
    return ok


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()[:16]


def run_checks(train_steps: int = 5) -> bool:
    """전체 점검. 모두 통과하면 True."""
    _results.clear()
    t_start = time.time()

    # 1. 환경 -----------------------------------------------------------------
    _out("\n[1/5] 환경")
    import torch

    _out("  torch " + torch.__version__)

    using_compat = False
    try:
        import torchtext_compat

        using_compat = torchtext_compat.install_as_torchtext()
    except ImportError:
        _out("  (torchtext_compat 를 찾을 수 없음 - 프로젝트 폴더에서 실행하세요)")

    try:
        import torchtext
    except Exception as exc:  # noqa: BLE001
        _check("torchtext import", False, type(exc).__name__ + ": " + str(exc))
        return _summary(t_start)

    kind = "torchtext_compat (순수 파이썬 호환 모듈)" if using_compat else "진짜 torchtext 0.18"
    _check("torchtext import", True, "-> " + kind)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        cap = torch.cuda.get_device_capability(0)
        _out("  GPU: " + torch.cuda.get_device_name(0) + " (sm_" + str(cap[0]) + str(cap[1]) + ")")
    else:
        device = torch.device("cpu")
        _out("  GPU: 없음 (CPU로 검사)")
    gpu_path = using_compat and device.type == "cuda"
    _check("GPU + torchtext 동시 사용", gpu_path, "GPU 경로" if gpu_path else "이 환경은 CPU 경로 (정상)")

    from torchtext.data.utils import get_tokenizer
    from torchtext.datasets import AG_NEWS, Multi30k
    from torchtext.vocab import build_vocab_from_iterator

    # 2. 토크나이저 ------------------------------------------------------------
    _out("\n[2/5] 토크나이저")
    basic = get_tokenizer("basic_english")
    _check("basic_english", basic("You can't <br />install TorchText; using pip (v2.0)!") == EXPECTED["tok_basic"])
    _check("None (공백 분리)", get_tokenizer(None)("a b  c") == ["a", "b", "c"])
    _check("callable 통과", get_tokenizer(str.upper)("ab") == "AB")

    try:
        tok = {
            "de": get_tokenizer("spacy", language="de_core_news_sm"),
            "en": get_tokenizer("spacy", language="en_core_web_sm"),
        }
    except Exception as exc:  # noqa: BLE001
        _check("spacy 토크나이저", False, type(exc).__name__ + ": " + str(exc))
        return _summary(t_start)

    # 3. 데이터셋 --------------------------------------------------------------
    _out("\n[3/5] 데이터셋 (없으면 자동 다운로드)")
    train = list(Multi30k(root=DATA_ROOT, split="train", language_pair=("de", "en")))
    valid = list(Multi30k(root=DATA_ROOT, split="valid", language_pair=("de", "en")))
    train = [p for p in train if p[0].strip() or p[1].strip()]
    valid = [p for p in valid if p[0].strip() or p[1].strip()]
    _check(
        "Multi30k 문장 수",
        [len(train), len(valid)] == EXPECTED["multi30k_counts"],
        "train " + str(len(train)) + " / valid " + str(len(valid)),
    )
    _check("Multi30k 내용 해시", _hash(train + valid) == EXPECTED["multi30k_hash"])
    _out("    de: " + train[0][0])
    _out("    en: " + train[0][1])

    ag = list(AG_NEWS(root=DATA_ROOT, split="train"))
    _check(
        "AG_NEWS",
        len(ag) == EXPECTED["ag_news_len"] and ag[0][0] == EXPECTED["ag_news_first"],
        str(len(ag)) + "건, 첫 라벨 " + str(ag[0][0]),
    )

    _check("spacy de 토큰화", len(tok["de"](train[0][0])) == EXPECTED["tok_de_len"])
    _check("spacy en 토큰화", tok["en"](train[0][1]) == EXPECTED["tok_en"])

    # 4. Vocab ----------------------------------------------------------------
    _out("\n[4/5] Vocab")
    specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
    UNK, PAD, BOS, EOS = range(4)
    V = {}
    for i, lang in enumerate(["de", "en"]):
        v = build_vocab_from_iterator(
            (tok[lang](p[i]) for p in train), min_freq=2, specials=specials, special_first=True
        )
        v.set_default_index(UNK)
        V[lang] = v
        _check(lang + " vocab 크기", len(v) == EXPECTED["vocab_" + lang + "_len"], str(len(v)))
        _check(lang + " vocab 순서 해시", _hash(v.get_itos()) == EXPECTED["vocab_" + lang + "_hash"])

    v = V["de"]
    _check("specials 위치", v.get_itos()[:4] == EXPECTED["vocab_head"])
    _check("OOV -> default index", v["존재하지않는토큰"] == UNK)
    _check("vocab(토큰리스트) 호출", v(["<bos>", "<eos>"]) == [BOS, EOS])
    _check("lookup_tokens 역변환", v.lookup_tokens([0, 1, 2, 3]) == specials)
    _check("__contains__", ("<bos>" in v) and ("존재하지않는토큰" not in v))

    # 5. 학습 루프 -------------------------------------------------------------
    _out("\n[5/5] 학습 (" + device.type + ")")
    from torch import nn
    from torch.utils.data import DataLoader

    def to_ids(text, lang):
        return torch.tensor([BOS] + V[lang](tok[lang](text)) + [EOS])

    def collate(batch):
        src = nn.utils.rnn.pad_sequence([to_ids(s, "de") for s, _ in batch], padding_value=PAD)
        tgt = nn.utils.rnn.pad_sequence([to_ids(t, "en") for _, t in batch], padding_value=PAD)
        return src.to(device), tgt.to(device)

    loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)

    class Seq2Seq(nn.Module):
        def __init__(self, n_src, n_tgt, d=128):
            super().__init__()
            self.src_emb, self.tgt_emb = nn.Embedding(n_src, d), nn.Embedding(n_tgt, d)
            self.transformer = nn.Transformer(d, 8, 2, 2, 256)
            self.fc = nn.Linear(d, n_tgt)

        def forward(self, src, tgt):
            mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0), device=tgt.device)
            return self.fc(self.transformer(self.src_emb(src), self.tgt_emb(tgt), tgt_mask=mask))

    model = Seq2Seq(len(V["de"]), len(V["en"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    losses = []
    t0 = time.time()
    src = tgt = None
    for step, (src, tgt) in enumerate(loader):
        out = model(src, tgt[:-1])
        loss = crit(out.reshape(-1, out.size(-1)), tgt[1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step + 1 >= train_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    _check(
        "DataLoader + collate",
        src.dim() == 2 and tgt.dim() == 2,
        "배치 src" + str(tuple(src.shape)) + " tgt" + str(tuple(tgt.shape)),
    )
    _check("텐서가 올바른 장치에", src.device.type == device.type, str(src.device))
    finite = all(l == l for l in losses)  # NaN 이면 False
    _check(
        "순전파+역전파",
        len(losses) == train_steps and finite,
        "loss {:.3f} -> {:.3f}".format(losses[0], losses[-1]),
    )
    detail = "    {} steps, {:.1f}s ({:.2f}s/step)".format(train_steps, elapsed, elapsed / train_steps)
    if device.type == "cuda":
        detail += ", GPU {:.0f} MB".format(torch.cuda.max_memory_allocated() / 1e6)
    _out(detail)

    return _summary(t_start)


def _summary(t_start):
    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    _out("\n" + "=" * 56)
    if passed == total:
        _out("전체 통과: {}/{}  ({:.1f}s)".format(passed, total, time.time() - t_start))
    else:
        _out("실패 {}건 / 전체 {}건  ({:.1f}s)".format(total - passed, total, time.time() - t_start))
        for name, ok in _results:
            if not ok:
                _out("  - " + name)
    _out("=" * 56)
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if run_checks() else 1)
