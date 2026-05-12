import torch


class MIResultsStore:
    """
    Storage for membership inference results, keyed by binary mask tensors.

    Maps each mask (stored as raw bytes) to a boolean annotation tensor.
    """

    def __init__(self):
        self._store = {}  # dict: mask_bytes → annotation_tensor (bool)

    def _tensor_to_key(
            self,
            tensor: torch.Tensor
    ) -> bytes:
        return tensor.to(dtype=torch.uint8).cpu().numpy().tobytes()

    def add(
            self,
            mask: torch.Tensor,
            annotation: torch.Tensor
    ) -> None:
        key = self._tensor_to_key(mask)
        self._store[key] = annotation.clone()

    def get(
            self,
            mask: torch.Tensor
    ) -> torch.Tensor | None:
        key = self._tensor_to_key(mask)
        return self._store.get(key)

    def items(self):
        """ Yield (mask, annotation) pairs with masks decoded back to bool tensors.
        """
        for key_bytes, annotation in self._store.items():
            mask_array = torch.frombuffer(key_bytes, dtype=torch.uint8)
            yield mask_array.bool(), annotation

    def get_results(self):
        """ Yield (train_annotation, test_annotation) pairs for each stored mask.

        train_annotation — the stored boolean annotation;
        test_annotation — elements in mask that are True but not in annotation (mask XOR annotation).
        """
        for key_bytes, annotation in self._store.items():
            mask_array = torch.frombuffer(key_bytes, dtype=torch.uint8)
            yield annotation, mask_array.bool() & (mask_array.bool() ^ annotation)
