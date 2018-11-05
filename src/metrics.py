import numpy as np
import sklearn.metrics as met

def get_recall(Y_true, Y_pred):
    """
    Calcul le recall correspond à la prediction Y_pred
    """
    return met.recall_score(Y_pred, Y_pred, average='macro')

def get_precision(Y_true, Y_pred):
    """
    Calcul la precision d'un modèle grace à une prediction
    """
    return 0

def get_accuracy(Y_true, Y_pred):
    """
    calcul l'exactitude d'un modèle grace à une prediction
    """
    return 0

def get_confusion_matrix(Y_true, Y_pred):
    """
    calcul la matrice de confusion d'un prédiction
    """
    pass

def plot_roc(Y_true, Y_pred):
    """
    trace la courbe roc d'une prediction
    """
    pass
