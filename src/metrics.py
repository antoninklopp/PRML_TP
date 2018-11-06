import numpy as np
import sklearn.metrics as met

MACHINE_ENSIMAG=False
try:
	import matplotlib.pyplot as plt
except ImportError:
	print("Machine ENSIMAG")
	MACHINE_ENSIMAG = True

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
    return met.accuracy_score(Y_true, Y_pred)

def get_all_metric(Y_true, Y_pred):
    """
    renvoie un dictionnaire contenant toutes les metrique utile pour carracteriser un modele
    :return: un dictionnaire contenant les metrics
    """
    return {
        "recall": get_recall(Y_true, Y_pred),
        "precision": get_precision(Y_true, Y_pred),
        "accuracy": get_accuracy(Y_true, Y_pred)
           }

def get_confusion_matrix(Y_true, Y_pred):
    """
    calcul la matrice de confusion d'un prédiction
    """
    return met.confusion_matrix(Y_true, Y_pred)


def plot_roc(Y_true, Y_proba, save=False):
    """
    trace la courbe roc d'une prediction
    :param Y_proba: vecteur contentant les scores de chaque pixel
	:param (optionnal) save: Saves the curve. If set to false it shows the plot to the user
    """
    if MACHINE_ENSIMAG:
        print("Impossible to run this on Ensimag machines. Needs pyplot.")
        exit()

    fpr, tpr, thresholds = met.roc_curve(Y_true, Y_proba)
    roc_auc = met.roc_auc_score(Y_true, Y_proba)
    plt.close()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.50, 0.10,"AUC = "+ str(roc_auc))
    if save is True:
        plt.savefig("output/plot_roc.png")
    else:
        plt.show()

    

def plot_presion_recall_curve(Y_true, Y_proba, save=False):
    """
    trace la precision en fonction du rappel
    :param Y_true: vecteur contenant les bonnes prediction
    :param Y_proba: vecteur contenant les scores
	:param (optionnal) save: Saves the curve. If set to false it shows the plot to the user
    :return: rien trace une courbe
    """

    if MACHINE_ENSIMAG:
        print("Impossible to run this on Ensimag machines. Needs pyplot.")
        exit()

    precision, recall, a = met.precision_recall_curve(Y_true, Y_proba)
    auc = met.auc(recall, precision)
    plt.close()
    plt.plot(recall, precision)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.text(0.50, 0.10,"AUC = "+ str(auc))
    if save is True:
        plt.savefig("output/plot_pression_recall_curve.png")
    else:
        plt.show()
