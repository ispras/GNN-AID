class MIResultsStore:
    def init(self):
        self._store = {}  # dict: mask_bytes → annotation_tensor (bool)

    def _tensor_to_key(self, tensor: torch.Tensor) -> bytes:
        # Переводим булевый тензор на CPU, затем в байты
        return tensor.to(dtype=torch.uint8).cpu().numpy().tobytes()

    def add(self, mask: torch.Tensor, annotation: torch.Tensor):
        key = self._tensor_to_key(mask)
        self._store[key] = annotation.clone()  # желательно клонировать, чтобы избежать случайных изменений

    def get(self, mask: torch.Tensor) -> torch.Tensor | None:
        key = self._tensor_to_key(mask)
        return self._store.get(key)

    def items(self):
        # Внимание: маски возвращаются в виде тензоров через обратное преобразование
        for key_bytes, annotation in self._store.items():
            mask_array = torch.frombuffer(key_bytes, dtype=torch.uint8)
            yield mask_array.bool(), annotation

    def method_name(self):
        # return - разметка на трейн (аннотация), разметка на тест (mask array xor аннотация)
        pass