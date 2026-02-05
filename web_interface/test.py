from sklearn.metrics import normalized_mutual_info_score, jaccard_score

a = [1, 1, 1, 1, 1, 1]
b = [0, 0, 0, 0, 1, 1]
print(normalized_mutual_info_score(a, b))
print(jaccard_score(a, b))
