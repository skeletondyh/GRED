import numpy as np
from sklearn.metrics import average_precision_score

def map_nested_fn(fn):
    """
    Recursively apply fn to the key-value pairs of a nested dict.
    """
    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }
    return map_fn

def binary_operator_diag(element_i, element_j):
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, bu_i * a_j + bu_j

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
        from https://github.com/rampasek/GraphGPS/blob/main/graphgps/metrics_ogb.py#L31
    '''

    ap_list = []

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            ap = average_precision_score(y_true[is_labeled, i],
                                         y_pred[is_labeled, i])

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)
