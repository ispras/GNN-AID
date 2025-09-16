import torch
from torch import device

from data_structures.configs import ConfigPattern, ModelModificationConfig
from aux.utils import EXPLAINERS_INIT_PARAMETERS_PATH, EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH, POISON_ATTACK_PARAMETERS_PATH, EVASION_ATTACK_PARAMETERS_PATH
from explainers.explainers_manager import FrameworkExplainersManager

from models_builder.gnn_models import FrameworkGNNModelManager, Metric
from base.datasets_processing import DatasetManager
from models_builder.models_zoo import model_configs_zoo

from attacks.clga import CLGA

# Here we perform all actions same way as in multi_attack_pipeline.py
my_device = device('cuda' if torch.cuda.is_available() else 'cpu')
full_name = ("single-graph", "Planetoid", 'Cora')

dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
    full_name=full_name,
    dataset_ver_ind=0
)

gnn = model_configs_zoo(dataset=dataset, model_name='gin_gin')

manager_config = ConfigPattern(
    _config_class="ModelManagerConfig",
    _config_kwargs={
        "mask_features": [],
        "optimizer": {
            "_class_name": "Adam",
            "_config_kwargs": {},
        }
    }
)

steps_epochs = 200
gnn_model_manager = FrameworkGNNModelManager(
    gnn=gnn,
    dataset_path=results_dataset_path,
    manager_config=manager_config,
    modification=ModelModificationConfig(model_ver_ind=0, epochs=steps_epochs)
)

poison_attack_config = ConfigPattern(
    _class_name="CLGAAttack",
    _import_path=POISON_ATTACK_PARAMETERS_PATH,
    _config_class="PoisonAttackConfig",
    _config_kwargs={
        "num_epochs": 300
    }
)

fgsm_evasion_attack_config = ConfigPattern(
    _class_name="FGSM",
    _import_path=EVASION_ATTACK_PARAMETERS_PATH,
    _config_class="EvasionAttackConfig",
    _config_kwargs={
        "epsilon": 0.002 * 1,
        "is_feature_attack": True,
    }
)

gnn_model_manager.set_poison_attacker(poison_attack_config=poison_attack_config)
gnn_model_manager.set_evasion_attacker(evasion_attack_config=fgsm_evasion_attack_config)

dataset.train_test_split()

gnn_model_manager.epochs = gnn_model_manager.modification.epochs = 0
train_test_split_path = gnn_model_manager.train_model(gen_dataset=dataset, steps=200,
                                                      save_model_flag=False,
                                                      metrics=[Metric("F1", mask='train', average=None)])
# gnn_model_manager.train_model(gen_dataset=dataset, steps=200)

metric_loc = gnn_model_manager.evaluate_model(
    gen_dataset=dataset, metrics=[Metric("F1", mask='test', average='macro')])
print(metric_loc)

# Now after attack performed we can explain our model as it was done in 04 tutorial

# --- Explain model ---
# Here we can choose the desired explainer and set the parameters of the class constructor of our explainer
explainer_init_config = ConfigPattern(
    _class_name="GNNExplainer(torch-geom)",
    _import_path=EXPLAINERS_INIT_PARAMETERS_PATH,
    _config_class="ExplainerInitConfig",
    _config_kwargs={
    }
)
# Define an explainer manager class
explainer_GNNExpl = FrameworkExplainersManager(
    init_config=explainer_init_config,
    dataset=dataset, gnn_manager=gnn_model_manager,
    explainer_name='GNNExplainer(torch-geom)',
)
# Here we can specify the run parameters of the explainer, such as the number of the element from the dataset
# for which we want to get an explanation.
explainer_run_config = ConfigPattern(
    _config_class="ExplainerRunConfig",
    _config_kwargs={
        "mode": "local",
        "kwargs": {
            "_class_name": "GNNExplainer(torch-geom)",
            "_import_path": EXPLAINERS_LOCAL_RUN_PARAMETERS_PATH,
            "_config_class": "Config",
            "_config_kwargs": {
                "element_idx": 666
            },
        }
    }
)
# Now we can run the experiment and get an explanation of how the model works on the input data sample.
explanation = explainer_GNNExpl.conduct_experiment(explainer_run_config)