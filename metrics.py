import pandas as pd
from sklearn.metrics import accuracy_score
from scipy import stats


def __get_groups(data, label_name, positive_label, group_condition):
    query = '&'.join([str(k) + '==' + str(v)
                      for k, v in group_condition.items()])
    label_query = label_name + '==' + str(positive_label)
    unpriv_group = data.query(query)
    unpriv_group_pos = data.query(query + '&' + label_query)
    priv_group = data.query('~(' + query + ')')
    priv_group_pos = data.query('~(' + query + ')&' + label_query)
    return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos


def __compute_probs(data_pred, label_name, positive_label, group_condition):
    unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = __get_groups(data_pred, label_name, positive_label,
                                                                             group_condition)
    unpriv_group_prob = (len(unpriv_group_pos)
                         / len(unpriv_group))
    priv_group_prob = (len(priv_group_pos)
                       / len(priv_group))
    return unpriv_group_prob, priv_group_prob


def disparate_impact(data_pred: pd.DataFrame, group_condition: dict, pred_label_name: str, positive_label: int):
    
    '''
    data_pred: Dataframe with true and predicted values
    group_condition: Dictionary where the keys are the sensitive variables and the values are the values of the unprivileged group
    pred_label_name: Name of the predicted label
    positive_label: Positive value of the label
    '''

    unpriv_group_prob, priv_group_prob = __compute_probs(
        data_pred, pred_label_name, positive_label, group_condition)
    return min(unpriv_group_prob / priv_group_prob,
               priv_group_prob / unpriv_group_prob) if unpriv_group_prob != 0 and priv_group_prob != 0 else 0

def accuracy(df_pred: pd.DataFrame, true_label: str, pred_label: str):
    '''
    data_pred: Dataframe with true and predicted values
    true_label: Name of the true label
    pred_label: Name of the predicted label
    '''

    return accuracy_score(df_pred[true_label].values, df_pred[pred_label].values)

def hmean(metrics: list):
    '''
    metrics: List of metrics values
    '''
    stats.hmean(metrics)