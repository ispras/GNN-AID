from torch_geometric.nn import MessagePassing

from explainers.explainer import Explainer, finalize_decorator
from explainers.explanation import Explanation


class GSATExplainer(Explainer):
    name = 'GSAT'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        return\
            gen_dataset.is_multi() and\
            any(isinstance(m, MessagePassing) for m in model_manager.gnn.modules())

    def __init__(self, gen_dataset, model, device):
        Explainer.__init__(self, gen_dataset, model)

        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert self.gen_dataset.is_multi()
        assert mode == "local"
        idx = kwargs.pop('element_idx')

        self.graph_idx = idx
        graph = self.gen_dataset.dataset[self.graph_idx]
        self.x = graph.x
        self.edge_index = graph.edge_index
        self.pbar.reset(total=1) # just perform forward
        self.model(self.x, self.edge_index)
        self.raw_explanation = self.model.att
        self.pbar.update(1)
        self.pbar.close()

    def _finalize(self):
        mode = self._run_mode
        assert mode == "global"
        num_classes, num_prot_per_class, class_connection, prototype_graphs = self.raw_explanation

        meta = {
            "num_classes": num_classes,
            "num_prot_per_class": num_prot_per_class
        }
        data = {
            'class_connection': class_connection,
            'base_graphs': [],
            'nodes': [],
        }
        for i in range(num_prot_per_class * num_classes):
            data['base_graphs'].append(int(prototype_graphs[i].base_graph))
            data['nodes'].append([int(x) for x in prototype_graphs[i].coalition])

        self.explanation = Explanation(type='subgraph', local=True, meta=meta, data=data)

        # Remove unpickable attributes
        self.pbar = None
