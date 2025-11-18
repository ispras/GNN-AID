# Класс, где есть property
class OtherClass:
    @property
    def processed_file_names(self):
        return "file_from_other.pt"

# Твой класс
class MyDataset:
    def __init__(self, processed_file_names=None):
        if processed_file_names:
            self._processed_file_names = processed_file_names

    @property
    def processed_file_names(self):
        pf = getattr(self, "_processed_file_names", None)
        if pf is None:
            return "data.pt"
        if callable(pf):
            return pf()
        return pf


res = MyDataset(processed_file_names=lambda: OtherClass.processed_file_names.__get__(None, OtherClass))
print(res.processed_file_names)