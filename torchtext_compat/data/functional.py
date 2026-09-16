"""torchtext.data.functional 중 노트북에서 실제로 쓰는 것만."""

from torch.utils.data import Dataset

__all__ = ["to_map_style_dataset"]


class _MapStyleDataset(Dataset):
    def __init__(self, iter_data):
        self._data = list(iter_data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]


def to_map_style_dataset(iter_data):
    """iterable을 map-style Dataset으로 감쌉니다(원본과 동일하게 전부 메모리에 적재)."""
    return _MapStyleDataset(iter_data)
