from src.lab2.lab2 import *
from src.metrics.metrics import get_all_metric, plot_metrics
import matplotlib.pyplot as plt
import cv2

def get_metrics(numImg=50, scale=2, minNeigh=5, minSize=30, maxSize=200):
    """
    Testing the metrics by changing parameters

    All the parameters are the one used in the get_true_predicted_faces function. 

    :return: recall, precision, accuracy
    """
    infos_file_path = ROOT_PATH+"Images/WIDER/WIDER_train_faces.txt"

    true_masks, predicted_masks, scores, success = get_true_predicted_faces(infos_file_path, numImg, scale, minNeigh, minSize, maxSize)
    true_masks_flatten = np.concatenate([i.flatten() for i in true_masks])
    predicted_masks_flatten = np.concatenate([i.flatten() for i in predicted_masks])

    print(true_masks_flatten.shape)
    print(predicted_masks_flatten.shape)

    metrics = get_all_metric(true_masks_flatten, predicted_masks_flatten)

    return metrics["recall"], metrics["precision"], metrics["accuracy"], success

def test_scale():
    """
    Test the impact of the scale on metrics
    """
    recall = []
    precision = []
    accuracy = []
    range_parameter = []
    success = []
    for i in range(110, 250, 20):
        r, p, a, s = get_metrics(scale=i/100.0)
        recall.append(r); precision.append(p); accuracy.append(a); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "scale", recall, precision, accuracy, "scale_comparison.png")

    plt.close()
    plt.plot(range_parameter, success)
    plt.savefig("output/success_scale.png")
    plt.close()

def test_min_neigh():
    """
    Test the impact of the neighbours on metrics
    """
    recall = []
    precision = []
    accuracy = []
    range_parameter = []
    success = []
    for i in range(1, 20):
        r, p, a, s = get_metrics(minNeigh=i)
        recall.append(r); precision.append(p); accuracy.append(a); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "min neighbours", recall, precision, accuracy, "min_neigh_comparison.png")

    plt.close()
    plt.plot(range_parameter, success)
    plt.savefig("output/success_minNeigh.png")
    plt.close()

def test_min_size():
    """
    Test the impact of the neighbours on metrics
    """
    recall = []
    precision = []
    accuracy = []
    range_parameter = []
    success = []
    for i in range(20, 300, 10):
        r, p, a, s = get_metrics(minSize=i)
        recall.append(r); precision.append(p); accuracy.append(a); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "min size", recall, precision, accuracy, "min_size_comparison.png")

    plt.close()
    plt.plot(range_parameter, success)
    plt.savefig("output/success_minSize.png")
    plt.close()

def test_max_size():
    """
    Test the impact of the neighbours on metrics
    """
    recall = []
    precision = []
    accuracy = []
    range_parameter = []
    success = []
    for i in range(30, 300, 10):
        r, p, a, s = get_metrics(maxSize=i)
        recall.append(r); precision.append(p); accuracy.append(a); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "max size", recall, precision, accuracy, "max_size_comparison.png")

    plt.close()
    plt.plot(range_parameter, success)
    plt.savefig("output/success_maxSize.png")
    plt.close()


if __name__ == "__main__":
    test_scale()
    # test_min_size()
    # test_min_neigh()
    # test_max_size()