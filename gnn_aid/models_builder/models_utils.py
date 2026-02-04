from typing import Any, Optional, List, Callable, Union, Tuple
from typing import Callable

import sklearn.metrics
import torch
import torch.nn as nn
from torch import tensor
from torch.utils.hooks import RemovableHandle
from torch_geometric.nn import MessagePassing

from gnn_aid.datasets import GeneralDataset


def apply_message_gradient_capture(
        layer: Any,
        name: str
) -> None:
    """
    # Example how get Tensors
    # for name, layer in self.gnn.named_children():
    #     if isinstance(layer, MessagePassing):
    #         print(f"{name}: {layer.get_message_gradients()}")
    """
    original_message = layer.message
    layer.message_gradients = {}

    def capture_message_gradients(
            x_j: torch.Tensor,
            *args,
            **kwargs
    ):
        x_j = x_j.requires_grad_()
        if layer.training:
            return original_message(x_j=x_j, *args, **kwargs)

        def save_message_grad(
                grad: torch.Tensor
        ) -> None:
            layer.message_gradients[name] = grad.detach()

        x_j.register_hook(save_message_grad)
        return original_message(x_j=x_j, *args, **kwargs)

    layer.message = capture_message_gradients

    def get_message_gradients(
    ) -> dict:
        return layer.message_gradients

    layer.get_message_gradients = get_message_gradients


# def apply_attention(
#         layer: Any,
#         name: str
# ) -> None:
#     """Modifies the forward method of the given layer to include edge_atten handling."""
#     original_forward = layer.forward
#
#     def modified_forward(self: Any, *args, edge_atten: Optional[Tensor] = None, **kwargs) -> Tensor:
#         # Inject edge_atten into kwargs if it's provided
#         if edge_atten is not None:
#             kwargs['edge_atten'] = edge_atten
#
#         return original_forward(*args, **kwargs)
#
#     layer.forward = modified_forward.__get__(layer)


def apply_decorator_to_graph_layers(
        model: Any,
        dec_f: Callable = apply_message_gradient_capture
) -> None:
    # TODO Kirill add more options
    """
    Example how use this def
    apply_decorator_to_graph_layers(gnn)
    """
    for name, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            dec_f(layer, name)
        elif isinstance(layer, torch.nn.Module):
            apply_decorator_to_graph_layers(layer, dec_f)


def apply_attention_to_messages(
        model: Any,
        att: torch.Tensor
) -> List[RemovableHandle]:
    handlers = []
    for _, layer in model.named_children():
        if isinstance(layer, MessagePassing):
            handlers.append(layer.register_message_forward_hook(attention_message_hook(att, layer)))
        elif isinstance(layer, torch.nn.Module):
            new_handlers = apply_attention_to_messages(layer, att)
            handlers.extend(new_handlers)
    return handlers


def attention_message_hook(
        att: Optional[torch.Tensor],
        layer: torch.nn.Module
):
    if att is None:
        return lambda module, input, out: out
    else:
        if not hasattr(layer, 'add_self_loops') or not layer.add_self_loops:
            return lambda module, input, out: out * att[out.shape[0], :]  # TODO assert here?
        else:
            if hasattr(layer, 'heads'):
                return lambda module, input, out: out * att.view(att.shape[0], 1, 1)
            else:
                return lambda module, input, out: out * att


class EdgeMaskingWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_edges: int):
        super().__init__()
        self.model = model
        self.edge_mask = nn.Parameter(torch.ones(num_edges))  # [E], requires_grad=True

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                if hasattr(module, 'add_self_loops'):
                    module.add_self_loops = False
                module.register_message_forward_hook(self._make_mask_hook())

    def _make_mask_hook(self):
        def hook(module, inputs, message_output):
            # message_output: [E, F]
            return message_output * self.edge_mask.to(message_output.device).view(-1, 1)
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# ================================================
# ================ Metrics
# ================================================


def top_k_metric(true_edges, pred_edges, k, metric='precision') -> float:
    """
    Calculate precision@k, recall@k, or f1@k metric for edge prediction

    Precision@k = (number of correctly predicted edges in top-k) / k
    Recall@k = (number of correctly predicted edges in top-k) / (total true edges)
    F1@k = 2 * (Precision@k * Recall@k) / (Precision@k + Recall@k)

    Args:
        true_edges: torch.Tensor shape (2, N) - ground truth edges
        pred_edges: torch.Tensor shape (2, M) - predicted edges (M >= k)
        k: int - number of top predictions to evaluate
        metric: str - metric to calculate: 'precision', 'recall', or 'f1'

    Returns:
        score: float - metric value in range [0, 1]
    """
    if pred_edges.size(1) == 0 or k == 0 or true_edges.size(1) == 0:
        return 0.0

    # Take only top-k predictions
    k_actual = min(k, pred_edges.size(1))
    pred_edges_top_k = pred_edges[:, :k_actual]

    # Convert true edges to set for fast lookup
    true_set = set(zip(
        true_edges[0].cpu().tolist(),
        true_edges[1].cpu().tolist()
    ))
    pred_set = set(zip(
        pred_edges_top_k[0].cpu().tolist(),
        pred_edges_top_k[1].cpu().tolist()
    ))
    correct_predictions = len(true_set & pred_set)

    # Calculate requested metric
    if metric == 'precision':
        return correct_predictions / k_actual

    elif metric == 'recall':
        total_true = true_edges.size(1)
        return correct_predictions / total_true

    elif metric == 'f1':
        precision = correct_predictions / k_actual
        total_true = true_edges.size(1)
        recall = correct_predictions / total_true

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'precision', 'recall', or 'f1'")


class Metric:
    available_metrics = {
        'Accuracy': sklearn.metrics.accuracy_score,
        'F1': sklearn.metrics.f1_score,
        'BalancedAccuracy': sklearn.metrics.balanced_accuracy_score,
        'Recall': sklearn.metrics.recall_score,
        'Precision': sklearn.metrics.precision_score,
        'Jaccard': sklearn.metrics.jaccard_score,

        'AUC': sklearn.metrics.roc_auc_score,
    }
    edge_prediction_metrics = {
        'Precision@k': lambda *a, **k: top_k_metric(*a, **k, metric='precision'),
        'Recall@k': lambda *a, **k: top_k_metric(*a, **k, metric='recall'),
        'F1@k': lambda *a, **k: top_k_metric(*a, **k, metric='f1'),
    }

    @staticmethod
    def add_custom(
            name: str,
            compute_function: Callable
    ) -> None:
        """
        Register a custom metric.
        Example for accuracy:

        >>> Metric.add_custom('accuracy', lambda y_true, y_pred, normalize=False:
        >>>     int((y_true == y_pred).sum()) / (len(y_true) if normalize else 1))

        :param name: name to refer to this metric
        :param compute_function: function which computes metric result:
         f(y_true, y_pred, **kwargs) -> value
        """
        if name in Metric.available_metrics:
            raise NameError(f"Metric '{name}' already registered, use another name")
        Metric.available_metrics[name] = compute_function

    def __init__(
            self,
            name: str,
            mask: Union[str, List[bool], torch.Tensor],
            **kwargs
    ):
        """
        :param name: name to refer to this metric
        :param mask: 'train', 'val', 'test', or a bool valued list
        :param kwargs: params used in compute function
        """
        self._name = name
        self.mask = mask
        self.kwargs = kwargs

    def name(
            self
    ) -> str:
        """ Name including kwargs """
        res = self._name
        kwargs = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
        if len(kwargs) > 0:
            res += '{' + kwargs + '}'
        return res

    def __str__(
            self
    ) -> str:
        return self.name()

    def compute(
            self,
            y_true,
            y_pred
    ):
        if y_true.device != "cpu":
            y_true = y_true.cpu()
        if y_pred.device != "cpu":
            y_pred = y_pred.cpu()

        if self._name in Metric.available_metrics:
            func = Metric.available_metrics[self._name]
        elif self._name in Metric.edge_prediction_metrics:
            func = Metric.edge_prediction_metrics[self._name]
        else:
            raise NotImplementedError()

        return func(y_true, y_pred, **self.kwargs)

    def needs_logits(
            self
    ) -> bool:
        """ Whether the metric accepts logits as predictions not labels. """
        if self._name in ['AUC']:
            return True
        else:
            return False

    def needs_all_node_pairs(
            self
    ) -> bool:
        """  """
        if self._name in self.edge_prediction_metrics:
            return True
        else:
            return False

    @staticmethod
    def create_mask_by_target_list(
            y_true,
            target_list: List = None
    ) -> torch.Tensor:
        if target_list is None:
            mask = [True] * len(y_true)
        else:
            mask = [False] * len(y_true)
        for i in target_list:
            if 0 <= i < len(mask):
                mask[i] = True
        return tensor(mask)


class GNNConstructorError(Exception):
    def __init__(
            self,
            *args
    ):
        self.message = args[0] if args else None

    def __str__(
            self
    ):
        if self.message:
            return f"GNNConstructorError: {self.message}"
        else:
            return "GNNConstructorError has been raised!"


# ================================================
# ================ Additional torch layers
# ================================================


class Concat(nn.Module):
    """ Concatenation of tensors as a torch function. Convenient to use as a model layer.
    """
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dimension = dimension

    def forward(self, tensors):
        # Concatenate a list of tensors along the specified dimension
        return torch.cat(tensors, dim=self.dimension)


class DotProduct(nn.Module):
    """ Dot-product of 2 tensors as a torch function. Convenient to use as a model layer.
    """
    def __init__(self, dimension=-1):
        super(DotProduct, self).__init__()
        self.dimension = dimension

    def forward(self, x1, x2):
        # Dot product of 2 tensors
        return (x1 * x2).sum(dim=-1)


# ================================================
# ================ Edge prediction over all graph node pairs
# ================================================


def predict_top_k_edges(model, data, exclude_edges, k=100, use_faiss=True,
                        faiss_k_per_node=100, is_directed=True, remove_loops=True):
    """
    Predict top-k new edges with FAISS support for large graphs

    Args:
        model: trained GNN model
        data: PyG Data object with graph structure
        exclude_edges: torch.Tensor shape (2, num_edges) - edges to exclude from predictions
        k: number of top edges to return (globally)
        use_faiss: use FAISS for fast search (recommended for large graphs)
        faiss_k_per_node: how many candidates to search per node via FAISS
        is_directed: if False, normalize edges to (i,j) where i < j and remove duplicates
        remove_loops: if True, exclude self-loops (i,i) from predictions

    Returns:
        top_edges: torch.Tensor shape (2, k) - indices of top-k node pairs
        top_scores: torch.Tensor shape (k,) - scores for these pairs
    """
    model.eval()

    with torch.no_grad():
        # Get embeddings for all nodes
        h = model(data.x, data.edge_index)

        # Normalize embeddings (important for correct dot product)
        # h_norm = F.normalize(h, p=2, dim=1)
        h_norm = h

        num_nodes = h.size(0)

        # Create set of existing edges for fast lookup
        existing_set = set()
        if exclude_edges is not None and exclude_edges.size(1) > 0:
            # Normalize exclude_edges for undirected graphs
            if not is_directed:
                exclude_edges_normalized = _normalize_edges(exclude_edges)
            else:
                exclude_edges_normalized = exclude_edges

            existing_set = set(zip(
                exclude_edges_normalized[0].cpu().tolist(),
                exclude_edges_normalized[1].cpu().tolist()
            ))
            print(f"Excluding {len(existing_set)} existing edges")

            # Compute scores for sample of existing edges
            h_src_existing = h[exclude_edges[0]]
            h_dst_existing = h[exclude_edges[1]]
            sample_scores = model.decode(h_src_existing, h_dst_existing)
            sample_scores = sample_scores.sigmoid().cpu()

            print(f"Existing edges: {exclude_edges.size(1)} edges")
            print(f"Existing edges scores:")
            print(f"  Mean: {sample_scores.mean():.4f}")
            print(f"  Std: {sample_scores.std():.4f}")
            print(f"  Min: {sample_scores.min():.4f}")
            print(f"  Max: {sample_scores.max():.4f}")

        # Choose strategy: FAISS or full enumeration
        total_pairs = num_nodes * num_nodes
        if use_faiss and total_pairs > 100e6:
            print(f"Using FAISS for large graph: {num_nodes} x {num_nodes} = {total_pairs} pairs")
            return _predict_with_faiss(model, data, h_norm, existing_set, k,
                                       faiss_k_per_node, is_directed, remove_loops)
        else:
            print(
                f"Using full enumeration for small graph: {num_nodes} x {num_nodes} = {total_pairs} pairs")
            return _predict_full_enumeration(model, data, h_norm, existing_set, k,
                                             is_directed, remove_loops)


def _normalize_edges(edges):
    """
    Normalize edges for undirected graph: ensure i < j for each edge (i,j)

    Args:
        edges: torch.Tensor shape (2, num_edges)

    Returns:
        normalized_edges: torch.Tensor shape (2, num_edges) with i < j
    """
    src, dst = edges[0], edges[1]

    # Swap where src > dst
    mask = src > dst
    src_new = torch.where(mask, dst, src)
    dst_new = torch.where(mask, src, dst)

    return torch.stack([src_new, dst_new], dim=0)


def _deduplicate_edges(edges, scores):
    """
    Remove duplicate edges and keep only unique ones with highest scores

    Args:
        edges: torch.Tensor shape (2, num_edges)
        scores: torch.Tensor shape (num_edges,)

    Returns:
        unique_edges: torch.Tensor shape (2, num_unique)
        unique_scores: torch.Tensor shape (num_unique,)
    """
    if edges.size(1) == 0:
        return edges, scores

    # Create dictionary: edge_tuple -> (score, index)
    edge_dict = {}
    for idx in range(edges.size(1)):
        edge_tuple = (edges[0, idx].item(), edges[1, idx].item())
        score = scores[idx].item()

        # Keep edge with higher score
        if edge_tuple not in edge_dict or score > edge_dict[edge_tuple][0]:
            edge_dict[edge_tuple] = (score, idx)

    # Extract unique edges and their scores
    unique_indices = [v[1] for v in edge_dict.values()]
    unique_edges = edges[:, unique_indices]
    unique_scores = scores[unique_indices]

    # Re-sort by scores
    sorted_scores, sorted_indices = torch.sort(unique_scores, descending=True)
    unique_edges = unique_edges[:, sorted_indices]

    return unique_edges, sorted_scores


def _predict_with_faiss(model, data, h_norm, existing_set, k, faiss_k_per_node,
                        is_directed, remove_loops):
    """Prediction using FAISS for fast search"""
    import faiss

    num_nodes = h_norm.size(0)
    embedding_dim = h_norm.size(1)

    # Parameter: how many candidates to take per node
    # For undirected graphs, we need more candidates to account for deduplication
    if not is_directed:
        faiss_k_per_node = min(faiss_k_per_node * 2, num_nodes)
    else:
        faiss_k_per_node = min(faiss_k_per_node, num_nodes)

    # Build FAISS index for all nodes
    h_np = h_norm.cpu().numpy().astype('float32')

    # Use Inner Product (cosine similarity for normalized vectors)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(h_np)

    print(f"FAISS index built. Searching top-{faiss_k_per_node} candidates per node...")

    # Search top candidates for each node
    similarities, candidate_indices = index.search(h_np, faiss_k_per_node)

    # Collect all candidate pairs
    all_pairs = []
    for src_idx in range(num_nodes):
        for rank, dst_idx in enumerate(candidate_indices[src_idx]):
            dst_idx = int(dst_idx)

            # Remove loops if requested
            if remove_loops and src_idx == dst_idx:
                continue

            # Normalize edge for undirected graph
            if not is_directed:
                edge = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            else:
                edge = (src_idx, dst_idx)

            # Skip existing edges
            if edge in existing_set:
                continue

            # Save (src, dst, similarity)
            all_pairs.append((edge[0], edge[1], similarities[src_idx, rank]))

    print(f"Found {len(all_pairs)} candidate pairs after filtering")

    if len(all_pairs) == 0:
        print("Warning: No candidate pairs found!")
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    # Sort by similarity and take more than k to account for deduplication
    all_pairs.sort(key=lambda x: x[2], reverse=True)

    # For undirected graphs, we already normalized, so no duplicates
    # But we take more candidates to be safe
    top_pairs = all_pairs[:min(k * 2 if not is_directed else k, len(all_pairs))]

    # Now compute exact scores via model for final candidates
    h = model(data.x, data.edge_index)

    final_src = torch.tensor([p[0] for p in top_pairs], dtype=torch.long, device=h.device)
    final_dst = torch.tensor([p[1] for p in top_pairs], dtype=torch.long, device=h.device)

    # Prepare embeddings for decode
    h_src_final = h[final_src]
    h_dst_final = h[final_dst]

    final_scores = model.decode(h_src_final, h_dst_final)
    final_scores = final_scores.sigmoid()

    # Create edges tensor
    edges = torch.stack([final_src, final_dst], dim=0)

    # Deduplicate if undirected (should already be normalized, but just in case)
    if not is_directed:
        edges, final_scores = _deduplicate_edges(edges, final_scores)

    # Take final top-k
    final_k = min(k, edges.size(1))
    top_edges = edges[:, :final_k]
    top_scores = final_scores[:final_k]

    print(f"Top-{final_k} edges found with scores from "
          f"{top_scores[-1]:.4f} to {top_scores[0]:.4f}")

    return top_edges.cpu(), top_scores.cpu()


def _predict_full_enumeration(model, data, h, existing_set, k, is_directed, remove_loops):
    """Prediction with full enumeration of all pairs (for small graphs)"""
    num_nodes = h.size(0)
    device = h.device

    # Request more candidates with margin for filtering
    k_candidates = k + (len(existing_set) if existing_set else 0)
    if not is_directed:
        k_candidates = k_candidates * 3  # More margin for undirected due to deduplication
    k_candidates = min(k_candidates * 2, num_nodes * num_nodes)

    # Store global top-k
    global_top_scores = None
    global_top_src = None
    global_top_dst = None

    # Process in batches by source nodes
    batch_size = 1000
    for src_start in range(0, num_nodes, batch_size):
        src_end = min(src_start + batch_size, num_nodes)
        batch_src_size = src_end - src_start

        # Create pairs for current batch of source nodes with all destination nodes
        src_batch = torch.arange(src_start, src_end, device=device)
        dst_nodes = torch.arange(num_nodes, device=device)

        src_repeated = src_batch.repeat_interleave(num_nodes)
        dst_repeated = dst_nodes.repeat(batch_src_size)

        # Filter before computing scores
        # Remove loops if requested
        if remove_loops:
            loop_mask = src_repeated != dst_repeated
            src_repeated = src_repeated[loop_mask]
            dst_repeated = dst_repeated[loop_mask]

        # For undirected graphs, only keep pairs where src <= dst to avoid duplicates
        if not is_directed:
            undirected_mask = src_repeated <= dst_repeated
            src_repeated = src_repeated[undirected_mask]
            dst_repeated = dst_repeated[undirected_mask]

        if len(src_repeated) == 0:
            continue

        # Prepare embeddings for decode
        h_src_batch = h[src_repeated]
        h_dst_batch = h[dst_repeated]

        # Compute scores for batch
        with torch.no_grad():
            batch_scores = model.decode(h_src_batch, h_dst_batch).sigmoid()

        # Take top-k from batch
        batch_k = min(k_candidates, len(batch_scores))
        batch_top_scores, batch_top_indices = torch.topk(batch_scores, batch_k)
        batch_top_src = src_repeated[batch_top_indices]
        batch_top_dst = dst_repeated[batch_top_indices]

        # Merge with global top-k
        if global_top_scores is None:
            global_top_scores = batch_top_scores
            global_top_src = batch_top_src
            global_top_dst = batch_top_dst
        else:
            # Concatenate and take top-k from combined results
            combined_scores = torch.cat([global_top_scores, batch_top_scores])
            combined_src = torch.cat([global_top_src, batch_top_src])
            combined_dst = torch.cat([global_top_dst, batch_top_dst])

            keep_k = min(k_candidates, len(combined_scores))
            top_scores, top_indices = torch.topk(combined_scores, keep_k)

            global_top_scores = top_scores
            global_top_src = combined_src[top_indices]
            global_top_dst = combined_dst[top_indices]

        print(f"Processed {src_end}/{num_nodes} source nodes, "
              f"current top score: {global_top_scores[0]:.4f}")

    # Now filter existing edges from final candidates
    if existing_set:
        mask = torch.tensor([
            (s.item(), d.item()) not in existing_set
            for s, d in zip(global_top_src.cpu(), global_top_dst.cpu())
        ], device=device)

        global_top_scores = global_top_scores[mask]
        global_top_src = global_top_src[mask]
        global_top_dst = global_top_dst[mask]

    # Create edges tensor
    edges = torch.stack([global_top_src, global_top_dst], dim=0)

    # For undirected graphs, edges should already be normalized (src <= dst)
    # But deduplicate just in case
    if not is_directed:
        edges, global_top_scores = _deduplicate_edges(edges, global_top_scores)

    # Final top-k
    final_k = min(k, edges.size(1))
    if final_k < k:
        print(f"Warning: Only {final_k} unique pairs available, returning all")

    top_edges = edges[:, :final_k]
    top_scores = global_top_scores[:final_k]

    print(f"Top-{final_k} edges found with scores from {top_scores[-1]:.4f} to {top_scores[0]:.4f}")

    return top_edges.cpu(), top_scores.cpu()


def mask_to_tensor(
        gen_dataset: GeneralDataset,
        mask: Union[str, int, Tuple[int], list, torch.Tensor] = 'test'
) -> torch.Tensor:
    """
    Convert a mask over nodes/edges/graphs to tensor of specific size.
    Mask can be 'train', 'val', 'test', 'all', or id, or a list of ids, or a tensor.

    :param gen_dataset: dataset
    :param mask: part of the dataset on which the output will be obtained.
     Can be a node id, graph id, or edge as a tuple (i,j).
     Can be string: 'train', 'val', 'test', 'all'.
     Can be Tensor of specific nodes/edges/graphs.
    :return: tensor of nodes/edges/graphs
    """
    task = gen_dataset.dataset_var_config.task

    if isinstance(mask, str):
        mask_tensor = {
            'train': gen_dataset.train_mask,
            'val': gen_dataset.val_mask,
            'test': gen_dataset.test_mask,
            'all': tensor([True] * len(gen_dataset.labels)),
        }[mask]

    elif isinstance(mask, torch.Tensor):
        mask_tensor = mask

    elif task.is_node_level():  # Node id
        assert not gen_dataset.is_multi()
        mask_tensor = tensor([False] * gen_dataset.info.nodes[0])
        mask_tensor[mask] = True  # for int or list of ints

    elif task.is_graph_level():  # Graph id
        assert gen_dataset.is_multi()
        mask_tensor = tensor([False] * len(gen_dataset.info.nodes))
        mask_tensor[mask] = True  # for int or list of ints

    elif task.is_edge_level():  # Edge
        # isinstance(mask, tuple)
        raise NotImplementedError

    else:
        raise RuntimeError(f"Cannot infer mask tensor for given mask {mask}.")

    return mask_tensor
