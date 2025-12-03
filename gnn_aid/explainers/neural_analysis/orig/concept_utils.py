# take a set of neuron concepts and pick the best one based on the concept divergence
# score. returns a dictionary of cleaned concepts and a set of concept objects.
def clean_concepts(neuron_concepts):
    cleaned_concepts = {}
    for neuron_idx, dic in neuron_concepts.items():
        top = sorted([(k, v) for (k, v) in dic.items()], key=lambda x: -x[1][1])
        val = top[0][1][1]
        i = 0
        best_obj = th = None
        best_obj_name = ''
        while i < len(top) and top[i][1][1] == val:
            if best_obj is None or top[i][1][0].length() < best_obj.length():
                best_obj = top[i][1][0]
                th = top[i][1][2]
                best_obj_name = top[i][0]
            i += 1
        cleaned_concepts[neuron_idx] = (best_obj, (val, th, best_obj_name))
    cleaned_concepts = {k: v for (k, v) in sorted(list(cleaned_concepts.items()), key=lambda x: x[1][1], reverse=True)}
    distilled = []
    for k, v in cleaned_concepts.items():
        if v[0] is None:
            continue
        if not any([v[0].name() == conc.name() for conc in distilled]):
            distilled.append(v[0])
    return cleaned_concepts, distilled


# get the highest scoring concepts for a given neuron based on divergence score
def get_top_concepts_for_neuron(neuron_concepts, n):
    return sorted([(k, v[1]) for (k, v) in neuron_concepts[n].items()], key=lambda x: -x[1])


# get the best concepts for all neurons
def get_best_concepts(neuron_concepts):
    return sorted([[n, sorted([v for (k, v) in neuron_concepts[n].items()], key=lambda x: -x[1])[0]] for n in
                   neuron_concepts.keys()], key=lambda x: -x[1][1])


# given a concept, return the neurons which match it, ordered by the divergence score
def get_top_neurons_for_concept(neuron_concepts, name):
    top = sorted(list(neuron_concepts.keys()), key=lambda x: neuron_concepts[x][name][1], reverse=True)
    for i in range(32):
        print(f'{top[i]} {neuron_concepts[top[i]][name][1]}')


# get the thresholds for the detected concepts
def get_ths(cleaned_concepts):
    ths = [0 for _ in range(64)]
    for k, (_, u) in cleaned_concepts.items():
        ths[k] = u[1]
    return ths


# get the names of the detected concepts
def get_names(cleaned_concepts):
    names = ['' for _ in range(64)]
    for k, (_, u) in cleaned_concepts.items():
        names[k] = u[2]
    return names


# get the divergence scores of the detected concepts
def get_scores(cleaned_concepts):
    ths = [0 for _ in range(64)]
    for k, (_, u) in cleaned_concepts.items():
        ths[k] = u[0]
    return ths
