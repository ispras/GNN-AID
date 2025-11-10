from datasets.datasets_manager import DatasetManager

dataset_mg_powergraph, _, results_dataset_path_mg_powergraph = DatasetManager.get_by_full_name(
    full_name=("multiple-graphs", "PowerGraph", "Cascades_ieee24",),
    dataset_ver_ind=0
)