from explainers.explainer import Explainer, finalize_decorator
from explainers.explanation import Explanation


class ProtExplainer(Explainer):
    name = 'ProtGNN'

    @staticmethod
    def check_availability(gen_dataset, model_manager):
        """ Availability check for the given dataset and model manager. """
        return\
            gen_dataset.is_multi() and\
            {'prot_layer_name'}.issubset(dir(model_manager.gnn))

    def __init__(self, gen_dataset, model, device):
        Explainer.__init__(self, gen_dataset, model)

        if hasattr(self.model, 'eval'):
            self.model.eval()
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

    @finalize_decorator
    def run(self, mode, kwargs, finalize=True):
        assert mode == "global"
        assert self.gen_dataset.is_multi()
        # Just get prototypes from the model
        self.pbar.reset(total=1)
        self.raw_explanation = getattr(self.model, self.model.prot_layer_name)\
            .result_prototypes(self.model.best_prots, True)
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

        self.explanation = Explanation(type='prototype', local=False, meta=meta, data=data)

        # Remove unpickable attributes
        self.pbar = None
