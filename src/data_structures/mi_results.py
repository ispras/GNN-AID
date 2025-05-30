import torch

class MIResultsStore:
    def __init__(self):
        self._store = {}  # dict: mask_bytes → annotation_tensor (bool)

    def _tensor_to_key(self, tensor: torch.Tensor) -> bytes:
        return tensor.to(dtype=torch.uint8).cpu().numpy().tobytes()

    def add(self, mask: torch.Tensor, annotation: torch.Tensor):
        key = self._tensor_to_key(mask)
        self._store[key] = annotation.clone()

    def get(self, mask: torch.Tensor) -> torch.Tensor | None:
        key = self._tensor_to_key(mask)
        return self._store.get(key)

    def items(self):
        for key_bytes, annotation in self._store.items():
            mask_array = torch.frombuffer(key_bytes, dtype=torch.uint8)
            yield mask_array.bool(), annotation

    def get_results(self):
        # return - разметка на трейн (аннотация), разметка на тест (mask array xor аннотация)
        for key_bytes, annotation in self._store.items():
            mask_array = torch.frombuffer(key_bytes, dtype=torch.uint8)
            yield annotation, mask_array.bool() & (mask_array.bool() ^ annotation)