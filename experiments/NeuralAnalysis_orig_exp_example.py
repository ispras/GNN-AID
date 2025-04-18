import torch_geometric
from torch_geometric.data import Data

import importlib
from explainers.neural_analysis.orig import vis, concept_utils
from explainers.neural_analysis.orig.concept_utils import *
from explainers.neural_analysis.orig.pipeline import load_dataset
from explainers.neural_analysis.orig.pipeline import train_standard_model
from explainers.neural_analysis.orig.concept_ranker import by_weight

print(torch_geometric.__version__)

def strip(concepts):
    lightweight = {}
    for k, dic in concepts.items():
        dic_ = {}
        for k_, (_, v1, v2) in dic.items():
            dic_[k_] = (None, v1, v2)
        lightweight[k] = dic_
    return lightweight

train_loader_mutag, test_loader_mutag, _, dataset_mutag, train_dataset, test_dataset, _ = load_dataset('MUTAG')
model_mutag = train_standard_model('MUTAG', 'GIN', fold=0)
neuron_concepts = model_mutag.concept_search('MUTAG', train_dataset, depth=2, top=64, augment=False, level=1)

# train_loader_mutag, test_loader_mutag, _, dataset_mutag, train_dataset, test_dataset, _ = load_dataset('PROTEINS')
# model_mutag = train_standard_model('PROTEINS', 'GIN', fold=0)
# neuron_concepts = model_mutag.concept_search('PROTEINS', train_dataset, depth=3, top=64, augment=False, level=1)

# train_loader_mutag, test_loader_mutag, _, dataset_mutag, train_dataset, test_dataset, _ = load_dataset('MUTAGENICITY')
# model_mutag = train_standard_model('MUTAGENICITY', 'GIN', fold=0)
# neuron_concepts = model_mutag.concept_search('MUTAGENICITY', train_dataset, depth=2, top=64, augment=False, level=1)

cleaned_concepts, distilled = clean_concepts(neuron_concepts)  # QUE What is distilled ? ? ?

units = [22, 0, 50]
graphs = [11, 61, 151]
for g_ in graphs:
    for n in units:
        g = dataset_mutag.get(g_)
        final_mask, node_values = model_mutag.expl_gp_neurons(g, 1, debug=True, rank={n}, gamma=1000,
                                                              sigma=get_ths(cleaned_concepts),
                                                              names=get_names(cleaned_concepts),
                                                              scores=get_scores(cleaned_concepts), cum=True,
                                                              show_labels=False, show_node_mask=True, explore=True,
                                                              as_molecule=True, show_contribs=True, force=True)
        vis.show_graph(Data(g.x, g.edge_index, edge_attr=g.edge_attr), final_mask,
                       node_values=None, show_labels=False,
                       anchor=get_ths(cleaned_concepts)[n] * 2,
                       as_molecule=True,
                       custom_name=f'mutag_concepts/mutag-graph{g_}_neuron{n}.png')


importlib.reload(concept_utils)

target_class = 1
neurons, vals = by_weight(model_mutag, target_class)
task = 'mutag'
dev = model_mutag.device


def get_global_vis(n):
    best_g = None
    best_s = float('-inf')
    for i in range(len(train_dataset)):
        g = train_dataset[i]
        pf = model_mutag.partial_forward(g.x.to(dev), g.edge_index.to(dev)).detach().cpu()
        score = pf[:, n].max()
        if score > best_s:
            best_s = score
            best_g = g
    return best_g, score


seen_concepts = set()

for n, v in zip(neurons, vals):
    if v <= 0:
        break
    if n in cleaned_concepts:
        if cleaned_concepts[n][1][2] in seen_concepts:
            continue

        g, _ = get_global_vis(n)
        final_mask, node_values = model_mutag.expl_gp_neurons(g, target_class, debug=True, rank={n}, gamma=1000,
                                                              sigma=get_ths(cleaned_concepts),
                                                              names=concept_utils.get_names(cleaned_concepts),
                                                              scores=get_scores(cleaned_concepts), cum=True,
                                                              show_labels=False, show_node_mask=True, explore=True,
                                                              as_molecule=True, show_contribs=True, force=True)
        vis.show_graph(Data(g.x, g.edge_index,
                            edge_attr=g.edge_attr), final_mask, node_values=node_values, show_labels=False,
                       anchor=get_ths(cleaned_concepts)[n] * 3, as_molecule=True,
                       custom_name=f'{task}_global/{task}_global_class{target_class}_neuron{n}_{v : .4f}.png')
        seen_concepts.add(cleaned_concepts[n][1][2])