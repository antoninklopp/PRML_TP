import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt


def get_recall(Y_true, Y_pred):
    """
    Calcul le recall correspond à la prediction Y_pred
    """
    return met.recall_score(Y_pred, Y_pred, average='macro')

def get_precision(Y_true, Y_pred):
    """
    Calcul la precision d'un modèle grace à une prediction
    """
    return met.precision_score(Y_true, Y_pred, average="macro")

def get_accuracy(Y_true, Y_pred):
    """
    calcul l'exactitude d'un modèle grace à une prediction
    """
    return met.accuracy_score(Y_true, Y_pred, average="macro")

def get_confusion_matrix(Y_true, Y_pred):
    """
    calcul la matrice de confusion d'un prédiction
    """
    return met.confusion_matrix(Y_true, Y_pred)

def plot_roc(Y_true, Y_proba):
    """
    trace la courbe roc d'une prediction
    """
    fpr, tpr, thresholds = met.roc_curve(Y_true, Y_proba)
    roc_auc = met.auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.50, 0.10,"AUC = "+str(roc_auc))
    plt.show()
    


def plot_presion_recall_cruve(Y_true, Y_proba):
    """
    trace la precision en fonction du rappel
    :param Y_true: vecteur contenant les bonnes prediction
    :param Y_proba: vecteur contenant les scores
    :return: rien trace une courbe
    """
    precision, recall, a = met.precision_recall_curve(Y_true, Y_proba)
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()
