import json
import numpy as np

explaind_inds = [0, 1, 4, 5, 6, 8, 11, 13, 14, 15, 16, 17, 18, 22, 24, 25, 26, 29, 1862, 2130]

# path = f"expl_No_Def"
# for n in explaind_inds:
#     for f in range(5):
#         with open(path + f"_{f}.json", "r") as fin:
#             res = json.load(fin)
#             node_res = res[n]

def iou(s1, s2):
    if not len(s1.union(s2)):
        return 0
    return len(s1.intersection(s2)) / len(s1.union(s2))

path_def = "/experiments/cora_reverse_0.01/expl_Def_1.json"
path_no_def = "/experiments/cora_reverse_0.01/v1/expl_No_Def_1.json"

with open(path_def, 'r') as fin:
    def_data = json.load(fin)

with open(path_no_def, 'r') as fin:
    no_def_data = json.load(fin)

no_def_data_set = {k: set(v.keys()) for k, v in no_def_data.items()}
def_data_set = {k: set(v.keys()) for k, v in def_data.items()}
out = []
for k in no_def_data.keys():
    out.append(iou(no_def_data_set[k], def_data_set[k]))
    print(iou(no_def_data_set[k], def_data_set[k]))

out = np.array(out)
print(np.mean(out), np.std(out))