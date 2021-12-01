import os, time, xgboost, warnings  
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def get_impotances(clf, X, y):
    clf = clf.fit(X, y)
    clf.feature_importances_
    results=pd.DataFrame()
    results['columns']=X.columns
    results['importances'] = clf.feature_importances_
    results.sort_values(by='importances',ascending=False,inplace=True)
    return results
    

def confusio_matrix(clf, X, y):
    clf = clf.fit(X, y)
    predict = clf.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, predict).ravel()
    
    pred_neg = np.array([tn, fn])
    pred_pos = np.array([fp, tp])
    contingency_matrix = np.array([pred_neg, pred_pos])

    group_names = ['True Neg','False Neg','False Pos','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in contingency_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in contingency_matrix.flatten()/np.sum(contingency_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in  zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(contingency_matrix, annot=labels, fmt='', cmap='Blues')
    plt.show()
    
    AUC = round(roc_auc_score(y, predict), 3)
    
    result = [str(tn), str(fp), str(fn), str(tp), AUC]
    result =  pd.DataFrame(result).T
    result.columns = ['True Neg', 'False Pos', 'False Neg', 'True Pos', 'AUC']
    return result


def plot_roc_curve(clf, X, y):
    prob = clf.predict_proba(X)
    prob = prob[:, 1]
    fper, tper, thresholds = roc_curve(y, prob)
    
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()