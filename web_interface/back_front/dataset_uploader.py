import os
import shutil
from pathlib import Path
from typing import List, Set

from gnn_aid.aux import Declare
from gnn_aid.data_structures import DatasetConfig
from gnn_aid.datasets import DatasetInfo, KnownFormatDataset, gen_dataset, GeneralDataset


class UploadError(Exception):
    def __init__(
            self,
            *args
    ):
        self.message = args[0] if args else None

    def __str__(
            self
    ):
        if self.message:
            return f"UploadError: {self.message}"
        else:
            return "UploadError has been raised!"


class DatasetUploader:
    """
    Handles uploading datasets throw frontend. Validates data and register new user's datasets.
    """
    _cache = {}  # {upload_id -> DatasetUploader}

    @staticmethod
    def get(upload_id: str, upload_dir: str) -> 'DatasetUploader':
        """ Return an uploade for the given upload session
        """
        if not upload_id in DatasetUploader._cache:
            DatasetUploader._cache[upload_id] = DatasetUploader(upload_id, upload_dir)

        return DatasetUploader._cache[upload_id]

    def __init__(self, upload_id: str, upload_dir: str | Path):
        self.upload_id = upload_id
        self.upload_dir: Path = Path(upload_dir)

        self.files: Set[Path] = set()
        self._tmp_name: str = upload_id
        self._tmp_root, _ = Declare.dataset_root_dir(DatasetConfig(('tmp', self._tmp_name)))
        (self._tmp_root / 'raw').mkdir(parents=True, exist_ok=True)
        self._tmp_dataset: GeneralDataset = None
        self.metainfo: DatasetInfo = None

    def check(self, files: List[str], metainfo: dict=None):
        """ Handle updated files and/or metainfo
        """
        # todo remove when extend to many files
        if len(files) > 0:
            print('len(files) > 0, clear')
            self.files.clear()
            self.metainfo = None

        for file_info in files:
            path = Path(file_info["path"])
            self.files.add(path)

        if len(files) > 0:
            # Analyze files
            try:
                self._analyze_files()
            except UploadError as e:
                return [e.message, None]

        if self.metainfo is None:
            self._induce_metainfo()

        if metainfo is not None:
            changed = self.metainfo.update(metainfo)
            print('updated metainfo', self.metainfo.to_dict())
            # Check some not implemented things
            e = None
            if self.metainfo.hetero:
                e = "Heterographs are not supported yet"
            if self.metainfo.count != 1:
                e = "Only 1-graph datasets are supported"

            if e:
                return [e, self.metainfo.to_dict()]

        try:
            self.metainfo.check()
        except (ValueError, AssertionError, AttributeError) as e:
            import traceback
            print(traceback.print_exc())
            return [str(e) or "some error, see server logs", None]

        try:
            self._try_construct_dataset()
            return ["SUCCESS", self.metainfo.to_dict()]
        except UploadError as e:
            return [e.message, self.metainfo.to_dict()]

    def _analyze_files(self):
        """ Parse uploaded files and move them to tmp dir
        """
        all_files = list(self.files)
        if len(all_files) > 1:
            raise UploadError("Uploading more than 1 file is not supported yet")

        edges_file = all_files[0]
        ext = edges_file.name.split('.')[-1]
        if ext != 'ij':
            raise UploadError("Only edge list format is supported. File should have name '*.ij'")

        # Move edges
        # todo avoid copying
        shutil.copy(edges_file, self._tmp_root / 'raw' / 'edges.ij')

    def _induce_metainfo(self):
        """ Induce metainfo from files.
        """
        self.metainfo = DatasetInfo()

        self.metainfo.name = self._tmp_name
        self.metainfo.format = 'ij'
        self.metainfo.count = 1  # todo extend
        self.metainfo.directed = False
        self.metainfo.hetero = False
        self.metainfo.nodes = [8]
        self.metainfo.remap = True
        self.metainfo.labelings = {}
        self.metainfo.node_attributes = {
            "names": [],
            "types": [],
            "values": []
        }

    def _try_construct_dataset(self):
        """
        Check consistency between files and metainfo.
        Parse the files and try to construct a dataset.
        """
        # Construct a gen dataset
        self.metainfo.save(self._tmp_root / 'metainfo')
        try:
            self._tmp_dataset = KnownFormatDataset(
                dataset_config=DatasetConfig(('tmp', self._tmp_name)))
        except (ValueError, AssertionError) as e:
            import traceback
            print(traceback.print_exc())
            raise UploadError(str(e) or "some error, see server logs")

    def cancel(self):
        """ Drop all uploaded data
        """
        shutil.rmtree(self._tmp_root, ignore_errors=True)
        shutil.rmtree(self.upload_dir, ignore_errors=True)

    def submit(self):
        """ Move the dataset to target directory with a correct name. Remove tmp files
        """
        assert self._tmp_dataset
        assert self.metainfo

        dc = DatasetConfig(('uploaded', self.metainfo.name))
        root, files_paths = Declare.dataset_root_dir(dc)

        shutil.move(self._tmp_root, root)
        shutil.rmtree(self.upload_dir)
