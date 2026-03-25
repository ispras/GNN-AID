import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, List, Dict, Any

from tqdm import tqdm

from gnn_aid.aux import Declare
from gnn_aid.data_structures.configs import DatasetConfig, ConfigPattern
from gnn_aid.datasets import KnownFormatDataset
from gnn_aid.datasets.dataset_info import DatasetInfo


class CitHepPhDataset(
    KnownFormatDataset
):
    """
    CitHepPh dataset from SNAP
    """
    url_edges_snap = "https://snap.stanford.edu/data/cit-HepPh.txt.gz"

    def __init__(
            self,
            dataset_config: Union[DatasetConfig, ConfigPattern]
    ):
        """
        """
        self._create_if_not(dataset_config)

        super(CitHepPhDataset, self).__init__(dataset_config)

        self._node_attributes = {}

    def _create_if_not(self, dataset_config: DatasetConfig):
        """
        """
        root, files_paths = Declare.dataset_root_dir(dataset_config)
        if (root / 'metainfo').exists():
            return

        raw = root / 'raw'
        raw.mkdir(parents=True, exist_ok=True)

        # Download edges
        import gzip
        import shutil
        import urllib.request

        tmp_gz = raw / "temp_download.gz"
        tmp_out = raw / "temp_extracted"
        edges_path = raw / "edges.ij"
        if not edges_path.exists():
            try:
                print(f"Downloading edges from {CitHepPhDataset.url_edges_snap}...")
                # download
                with urllib.request.urlopen(CitHepPhDataset.url_edges_snap) as response, open(tmp_gz, "wb") as f:
                    shutil.copyfileobj(response, f)

                # extract
                with gzip.open(tmp_gz, "rb") as f_in, open(tmp_out, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

                # rename
                if edges_path.exists():
                    edges_path.unlink()
                tmp_out.rename(edges_path)

                print("Done.")
            finally:
                # cleanup
                if tmp_gz.exists():
                    tmp_gz.unlink()

        # Get ids
        ids = set()

        with open(edges_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                ids.add(parts[0])
                ids.add(parts[1])

        # Get attributes
        (raw / 'node_attributes').mkdir(exist_ok=True)
        attrs = ["authors", "title", "abstract"]

        fetch_all(
            article_ids=ids ,
            fields=attrs,
            batch_size=100,
            delay=3.1,
            output_dir=raw / 'node_attributes',
        )

        # Create DatasetInfo
        info = DatasetInfo()
        info.name = dc.full_name[-1]
        info.format = "ij"
        info.count = 1
        info.directed = True
        info.hetero = False
        info.nodes = [len(ids)]
        info.remap = True
        info.node_attributes = {
            "names": attrs,
            "types": ["other"]*len(attrs),
            "values": [None]*len(attrs)}
        info.edge_attributes = None
        info.labelings = {}

        info.save(root / 'metainfo')

    def node_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, 'torch.Tensor']]:
        """ Lazy get - reads attributes from file only when asked.
        """
        if attrs is None:
            attrs = sorted(self.info.node_attributes['names'])
        res = {}
        for a in attrs:
            if a not in self._node_attributes:
                with open(self.node_attributes_dir / a, 'r') as f:
                    attr_dict = json.load(f)
                self._node_attributes[a] = []
                node_attributes = {}
                for ix, orig in self._iter_nodes():
                    node_attributes[ix] = attr_dict.get(orig)
                self._node_attributes[a].append(node_attributes)

            res[a] = self._node_attributes[a]

        return res

    def _read_attributes(
            self
    ) -> None:
        pass  # Do not read all attributes

    def edge_attributes(
            self,
            attrs: List[str] = None
    ) -> Dict[str, Union[list, 'torch.Tensor']]:
        """ Get edge attributes as a dict {name -> list}"""
        return None

    def _attribute_to_feature(
            self,
            attr: str,
            value: Any
    ) -> List[float]:
        return [0, 0, 0]


# ArXiv OAI-PMH / export API base URL
ARXIV_API_URL = "https://export.arxiv.org/api/query"

# XML namespaces used by the Atom feed
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def pad_id(article_id: str | int) -> str:
    """Pad article ID to 7 digits with leading zeros."""
    return str(article_id).strip().zfill(7)


def build_query_url(padded_ids: list[str]) -> str:
    """Build an ArXiv API query URL for a batch of hep-ph IDs."""
    id_list = ",".join(f"hep-ph/{pid}" for pid in padded_ids)
    params = urllib.parse.urlencode({
        "id_list": id_list,
        "max_results": len(padded_ids),
    })
    return f"{ARXIV_API_URL}?{params}"


def fetch_batch(padded_ids: list[str], retries: int = 3, retry_delay: float = 60.0) -> dict:
    """
    Fetch metadata for a batch of padded IDs from ArXiv.
    Returns a dict: {padded_id -> {authors, title, abstract}}.
    """
    url = build_query_url(padded_ids)

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read()
            break
        except Exception as exc:
            print(f"  [!] Attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(retry_delay)
            else:
                print(f"  [!] All retries exhausted for batch {padded_ids}")
                return {}

    root = ET.fromstring(content)
    results = {}

    for entry in root.findall("atom:entry", NS):
        # Extract the canonical ID from the <id> tag, e.g.
        # http://arxiv.org/abs/hep-ph/0010005v2  ->  0010005
        raw_id_elem = entry.find("atom:id", NS)
        if raw_id_elem is None:
            continue
        raw_id = raw_id_elem.text.strip()
        # Get the numeric part after "hep-ph/"
        if "hep-ph/" in raw_id:
            numeric = raw_id.split("hep-ph/")[-1].split("v")[0]  # strip version
        else:
            numeric = raw_id.rsplit("/", 1)[-1].split("v")[0]

        authors = [
            author.find("atom:name", NS).text.strip()
            for author in entry.findall("atom:author", NS)
            if author.find("atom:name", NS) is not None
        ]

        title_elem = entry.find("atom:title", NS)
        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

        summary_elem = entry.find("atom:summary", NS)
        abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

        results[numeric] = {
            "authors": authors,
            "title": title,
            "abstract": abstract,
        }

    return results


def fetch_all(
        article_ids: list[str | int],
        fields: list[str],
        batch_size: int = 20,
        delay: float = 3.0,
        output_dir: Path = Path("."),
) -> None:
    """
    Fetch metadata for all IDs and save everything into a single JSON file as dict
     {id -> {field -> value}}.

    Args:
        article_ids: Raw IDs (will be padded to 7 digits).
        fields: Subset of ["authors", "title", "abstract"] to save.
        batch_size: How many IDs to request per API call.
        delay: Seconds to wait between batches.
        output_dir: Directory to put result file.
    """
    valid_fields = {"authors", "title", "abstract"}
    invalid_fields = [field for field in fields if field not in valid_fields]
    if invalid_fields:
        raise ValueError(f"Unsupported fields: {invalid_fields}. Allowed: {sorted(valid_fields)}")

    id_map = {str(aid).strip(): pad_id(aid) for aid in article_ids}
    original_ids = list(id_map.keys())
    padded_ids = [id_map[original_id] for original_id in original_ids]

    all_meta: dict[str, dict] = {}
    batches = [padded_ids[i:i + batch_size] for i in range(0, len(padded_ids), batch_size)]
    print(f"Fetching {len(padded_ids)} articles in {len(batches)} batch(es)...")

    for idx, batch in tqdm(list(enumerate(batches, 1))):
        batch_result = fetch_batch(batch)
        all_meta.update(batch_result)

        if idx < len(batches):
            time.sleep(delay)

    print(f"Fetched metadata for {len(all_meta)} articles.")

    node_info: dict = {}

    for orig_id, padded_id in id_map.items():
        entry = all_meta.get(padded_id)

        if entry is None:
            print(f"  [!] No data found for id={orig_id} (hep-ph/{padded_id})")
            node_info[orig_id] = {field: None for field in fields}
            continue

        node_info[orig_id] = {field: entry.get(field) for field in fields}

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "node_info"
    filename.write_text(
        json.dumps(node_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved node info -> {filename}")


if __name__ == '__main__':
    dc = DatasetConfig(('my', 'snap', "Cit-Hep-Ph"))
    dataset = CitHepPhDataset(dc)

    print(dataset.info.to_dict())

    # dvc = DatasetVarConfig(
    #     task=Task.EDGE_PREDICTION,
    #     features=FeatureConfig(),
    #     )
    # dataset.build(dvc)
