import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets_block.datasets_manager import DatasetManager
from explainers.explainers_manager import FrameworkExplainersManager
from models_builder.gnn_models import FrameworkGNNModelManager, ProtGNNModelManager, Metric
from models_builder.models_zoo import model_configs_zoo


# from datasets_block.datasets_processing import Datasets
# from course_work.prot_datasets import SentiGraphDataset


def test_prot(i=None, conv=None, batch_size=24, seed=5,
              percent_train_class: float = 0.6,
              percent_test_class: float = 0.4,
              clst=0.10, sep=0.5,
              save_thrsh=1.0, lr=0.05, dataset_name: str = "MUTAG"):
    """

    @param i: savename
    @param conv: convolution type
    @param batch_size:
    @param seed:
    @param data_split_ratio: [train/validation/test]
    @param clst: cluster loss coeff
    @param sep: separation loss coeff
    @param save_thrsh:
    @param lr:
    @return:
    """
    print('start loading data======================')
    """
    dataset, data, results_dataset_path = Datasets.get_pytorch_geometric(
        full_name=("Homogeneous", "TUDataset", 'MUTAG'),
        dataset_ver_ind=0)
    """

    # if dataset_name.lower() == 'mutag':
    #     dataset = TUDataset(root='./data/torch-geometric', name='MUTAG')
    # elif dataset_name.lower() == 'graph-sst2':
    #     dataset = SentiGraphDataset(root="/home/sazonov/PycharmProjects/Interpretation/src/experiments/datasets",
    #                                 name='Graph-SST2')
    # else:
    #     raise NotImplementedError

    # my_device = device('cuda' if is_available() else 'cpu')
    full_name = (LibPTGDataset.data_folder, "Homogeneous", "TUDataset", "MUTAG")
    dataset, data, results_dataset_path = DatasetManager.get_by_full_name(
        full_name=full_name,
        dataset_ver_ind=0
    )

    # data = dataset[0]

    data.x = data.x.float()
    dataset.train_test_split(percent_train_class=percent_train_class, percent_test_class=percent_test_class)

    # dataset = gen_dataset.dataset

    # input_dim = dataset.num_node_features
    # output_dim = int(dataset.num_classes)

    # num_train = int(data_split_ratio[0] * len(dataset))
    # num_eval = int(data_split_ratio[1] * len(dataset))
    # num_test = len(dataset) - num_train - num_eval

    # train_dataset = reduce(lambda x, y: x + y,
    #                        [dataset[n: n + 1] for n, x in enumerate(dataset.data.train_mask)
    #                         if x])
    # test_dataset = reduce(lambda x, y: x + y,
    #                       [dataset[n: n + 1] for n, x in enumerate(dataset.data.test_mask)
    #                        if x])

    # train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test])  # ,
    # generator=torch.Generator().manual_seed(seed))

    # print(train.indices)

    # dataloader = dict()
    # dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    # dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=False)
    # dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)

    print('start training model====================')
    # if conv == 'gcn':
    #     conv_classes = ('GCNConv', 'GCNConv', 'GCNConv')
    # elif conv == 'gin':
    #     conv_classes = ('GINConv', 'GINConv', 'GINConv')
    # model = GNNStructureProt(
    #     conv_classes=conv_classes,
    #     layers_sizes=(dataset.num_node_features, 128, 128, 128),
    #     num_classes=dataset.num_classes,
    #     activations=('torch.relu', 'torch.relu', 'torch.relu'),
    # )

    model = model_configs_zoo(dataset=dataset, model_name='gin_gin_gin_lin_lin_prot')

    """
    #Fixing model parameters
    ckpt_pth = os.path.join("/home/sazonov/PycharmProjects/Interpretation/models", "Protgnn.pth")
    if os.path.exists(ckpt_pth):
        print("LOADING MODEL")
        #update_state_dict(model, ckpt_pth)
        model.load_state_dict(torch.load(ckpt_pth))
        print("MODEL SUCCESSFULLY LOADED")
    else:
        print("SAVING MODEL WEIGHTS")
        torch.save(model.state_dict(), ckpt_pth)
        print("MODEL SAVED")
    torch.nn.init.normal(model.prototype_vectors) #Randomize prototype initialization here!
    print("PROTOTYPE VECTORS RANDOMIZED")
    """
    prot_gnn_mm = ProtGNNModelManager(gnn=model,
                                      dataset_path=results_dataset_path, )
    # TODO Misha use as training params: clst=clst, sep=sep, save_thrsh=save_thrsh, lr=lr
    best_acc = prot_gnn_mm.train_model(gen_dataset=dataset, steps=100,
                                       metrics=[Metric("F1", mask='train', average=None)])

    explainer_Prot = FrameworkExplainersManager(explainer_name='ProtGNN',
                                                dataset=dataset, gnn_manager=prot_gnn_mm,
                                                explainer_ver_ind=0, )
    explainer_Prot.conduct_experiment(mode='global')

    # explainer_result_file_path = "/home/sazonov/PycharmProjects/Interpretation/results/ProtGNN/explanation" + str(
    #     i) + ".json"
    # explanation = getattr(model, model.prot_layer_name).result_prototypes(best_prots, True)
    # explanation.create_json(explainer_result_file_path)
    return best_acc


def update_state_dict(model, state_dict):
    original_state_dict = model.state_dict()
    loaded_state_dict = dict()
    for k, v in state_dict.items():
        if k in original_state_dict.keys():
            loaded_state_dict[k] = v
    model.load_state_dict(loaded_state_dict)


def selection(dataset_name='MUTAG'):
    # this function is designed for hyper-parameters selection
    clst = np.array([0.0, 0.01, 0.05, 0.1, 0.5])
    sep = np.array([0.0, 0.01, 0.05, 0.1, 0.5])
    # print(sep.astype(str).tolist())
    '''

    scores = np.zeros((5, 5))
    stds = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            #if (i < 3 and j < 2):
            #    continue
            clst_i = clst[i]
            sep_j = sep[j]
            acc_lst = []
            for k in range(3):
                while True:
                    try:
                        res = test_prot(100*(i+1)+10*(j+1)+k, 'gin', data_split_ratio=[0.8, 0.1, 0.1], clst=clst_i, sep=sep_j, dataset_name=dataset_name)
                    except:
                        continue
                    else:
                        break
                acc_lst.append(res)
            scores[i][j] = np.mean(acc_lst)
            stds[i][j] = np.std(acc_lst)
    out = pd.DataFrame(scores, columns=clst)
    out.to_csv("/home/sazonov/PycharmProjects/Interpretation/results/ProtGNN/parameters.csv")
    pd.DataFrame(stds, columns=clst).to_csv("/home/sazonov/PycharmProjects/Interpretation/results/ProtGNN/stds.csv")
    #out = out.to_numpy()
    
    # print(stds)
    '''

    out = pd.read_csv(
        "/home/sazonov/PycharmProjects/Interpretation/results/ProtGNN/parameters.csv").to_numpy()
    out_stds = pd.read_csv(
        "/home/sazonov/PycharmProjects/Interpretation/results/ProtGNN/stds.csv").to_numpy()
    # print(scores)

    print(out)

    # float_formatter = "{:.2f}".format
    # np.set_printoptions(formatter={'float_kind': float_formatter})

    fig, ax = plt.subplots()
    im = ax.imshow(out)

    ax.set_xticks(np.arange(5), labels=clst.tolist())
    ax.set_yticks(np.arange(5), labels=sep.tolist())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, str(np.round(out[i, j], 3)) + ' +- \n' + str(
                np.round(out_stds[i, j], 3)),
                           ha="center", va="center", color="w")

    ax.set_title("Model score depending on clst/sep loss\nMean +- Std")
    fig.tight_layout()
    plt.show()

    """
    for i in range(10):
        test_prot(i, 'gin', data_split_ratio=[0.5, 0.5, 0.0])
    """


def test_with_thrsh(save_thsh=0.88):
    i = 0
    save_thrsh = 0.88
    while i < 20:
        # acc = test_prot(i, 'gin', data_split_ratio=[0.6, 0.4, 0.0], clst=0.1, sep=0.05, save_thrsh=0.90, lr=0.02) #for debug

        try:
            acc = test_prot(i, 'gin', data_split_ratio=[0.6, 0.4, 0.0], clst=0.01, sep=0.00,
                            save_thrsh=0.90, lr=0.03)
        except:
            continue
        if acc >= save_thrsh:
            i += 1


def test_mutag():
    i = 0
    result_acc = []
    clst = np.array([0.0, 0.01, 0.05, 0.1, 0.5])
    sep = np.array([0.0, 0.01, 0.05, 0.1, 0.5])
    while i < 10:
        # acc = test_prot(i, 'gin', data_split_ratio=[0.6, 0.4, 0.0], clst=0.00, sep=0.05, lr=0.03)
        try:
            acc = test_prot(i, 'gin', data_split_ratio=[0.6, 0.4, 0.0], clst=0.00, sep=0.00,
                            lr=0.03)
            if acc > 0.7:
                continue
            result_acc.append(acc)
        except:
            continue
        i += 1

    print("Mean accuracy", np.mean(result_acc))


def test_sst2():
    i = 100
    while i < 150:
        try:
            clst = np.random.uniform(0.1, 0.5)
            sep = np.random.uniform(0.1, 0.5)
            test_prot(i, 'gin', data_split_ratio=[0.8, 0.1, 0.1], clst=clst, sep=sep, lr=0.03,
                      dataset_name='Graph-SST2')
        except Exception as e:
            print(e)
            continue
        i += 1

    # print("Mean accuracy", np.mean(result_acc))
    # test_prot(0, 'gin', data_split_ratio=[0.8, 0.1, 0.1], clst=0.01, sep=0.00, lr=0.03, dataset_name='Graph-SST2')


if __name__ == "__main__":
    # test_mutag()
    # test_sst2()
    test_prot()
    # selection(dataset_name='Graph-SST2')
