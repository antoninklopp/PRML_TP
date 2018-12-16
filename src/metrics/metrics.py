import numpy as np
import sklearn.metrics as met
from src.metrics.overlapping import overlapping_predicted

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
    # return met.recall_score(Y_true, Y_pred, average='macro')
    print("recall", np.sum(Y_true * Y_pred)/np.sum(Y_true))
    if np.sum(Y_true) == 0:
        return 0
    return np.sum(Y_true * Y_pred)/np.sum(Y_true) 

def get_recall_rectangle(rectangles_true, rectangles_predicted):
    """
    Calcule le recall correspondant a la prediction. 
    Utilise les rectangles de tetes plutot que des masques 
    comme dans la fonction get_recall. 
    """
    Y_true = 0
    for r in rectangles_true:
        o = overlapping_predicted(r, rectangles_predicted, 0.5)
        if o is True:
            Y_true += 1

    # print(Y_true, len(rectangles_predicted), len(rectangles_true))

    return min(1, Y_true/len(rectangles_true))


def get_precision(Y_true, Y_pred):
    """
    Calcul la precision d'un modèle grace à une prediction
    """
    print("precision", np.sum(Y_true * Y_pred)/np.sum(Y_pred))
    if np.sum(Y_pred) == 0:
        return 0
    return np.sum(Y_true * Y_pred)/np.sum(Y_pred) 

def get_precision_rectangle(rectangles_true, rectangles_predicted):
    """
    Calcule la precision correspondant a la prediction. 
    Utilise les rectangles de tetes plutot que des masques 
    comme dans la fonction get_recall. 
    """
    Y_true = 0

    if len(rectangles_predicted) == 0:
        return 0

    for r in rectangles_true:
        o = overlapping_predicted(r, rectangles_predicted, 0.5)
        if o is True:
            Y_true += 1

    return min(1, Y_true/len(rectangles_predicted))


def get_ground_truth(true_rectangles, predicted_rectangles, threshold=0.5):
    y_true = []
    for image in range(len(true_rectangles)):
        for r in predicted_rectangles[image]:
            if r.shape[0] == 0:
                y_true.append(False)
            else:
                o = overlapping_predicted(r, true_rectangles[image], threshold)
                if o is True:
                    y_true.append(True)
                else:
                    y_true.append(False)
    return y_true


def get_accuracy(Y_true, Y_pred):
    """
    calcul l'exactitude d'un modèle grace à une prediction
    """
    TP = np.sum(Y_true * Y_pred)
    TN = np.sum(np.subtract(np.ones(Y_true.shape), Y_true) * np.subtract(np.ones(Y_pred.shape), Y_pred))
    print("accuracy", (TP + TN)/(Y_true.shape[0]))
    return (TP + TN)/(Y_true.shape[0])

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


def plot_roc(Y_true, Y_proba, name="output/plot_roc.png", save=False):
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
        plt.savefig(name)
    else:
        plt.show()



def plot_presion_recall_curve(Y_true, Y_proba, name="output/plot_pression_recall_curve.png", save=False):
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
        plt.savefig(name)
    else:
        plt.show()


def plot_metrics(range_parameter, parameter_name, recall, precision, accuracy=None, file_output="recall_accuracy_precision.png"):
    """
    Print the 2 differents metrics in one single pyplot

    :param range_parameter: a list with the range of the parameter we want to plot the metrics for
    :param paramater_name: name of the parameter to put its name on the curve
    :param recall, precision, accuracy: recall, precision, accuracy we want to plot for the parameter
    :param file_output: the path to the file we want to output to
    """

    if accuracy is None:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        fig.tight_layout(pad=10, w_pad=4, h_pad=4)

    axes[0].plot(range_parameter, recall, label=parameter_name)
    axes[0].set_xlabel(parameter_name)
    axes[0].set_ylabel("Recall")
    axes[0].set_title("Recall")

    axes[1].plot(range_parameter, precision, label=parameter_name)
    axes[1].set_xlabel(parameter_name)
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision")

    if accuracy is not None:
        axes[2].plot(range_parameter, accuracy, label=parameter_name)
        axes[2].set_xlabel(parameter_name)
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Accuracy")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20),  shadow=True, ncol=2)

    plt.savefig("output/" + file_output)
