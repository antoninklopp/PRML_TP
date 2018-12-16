from src.lab2.lab2 import *
from src.metrics.metrics import get_recall_rectangle, get_precision_rectangle, plot_metrics
import matplotlib.pyplot as plt
import cv2

def get_metrics(numImg=100, scale=2, minNeigh=5, minSize=30, maxSize=200):
    """
    Testing the metrics by changing parameters

    All the parameters are the one used in the get_true_predicted_faces function. 

    :return: recall, precision, accuracy
    """
    infos_file_path = ROOT_PATH+"Images/rectangle.txt"

    true_rectangles, predicted_rectangles, scores, success = get_true_predicted_rectangles(infos_file_path, numImg, scale, minNeigh, minSize, maxSize)

    recall = []
    precision = []

    print(len(true_rectangles), len(predicted_rectangles))

    for i in range(len(true_rectangles)):
        recall.append(get_recall_rectangle(true_rectangles[i], predicted_rectangles[i]))
        precision.append(get_precision_rectangle(true_rectangles[i], predicted_rectangles[i]))

    return np.mean(np.array(recall)), np.mean(np.array(precision)), success

def test_scale():
    """
    Test the impact of the scale on metrics
    """
    recall = []
    precision = []
    range_parameter = []
    success = []
    for i in range(101, 301, 2):
        print("current scale ", i/100.0)
        r, p, s = get_metrics(scale=i/100.0)
        recall.append(r); precision.append(p); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "scale", recall, precision, None, "scale_comparison.png")

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
    range_parameter = []
    success = []
    for i in range(1, 20):
        r, p, s = get_metrics(minNeigh=i)
        recall.append(r); precision.append(p); range_parameter.append(i); success.append(s)
    
    plot_metrics(range_parameter, "min neighbours", recall, precision, None, "min_neigh_comparison.png")

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
    range_parameter = []
    success = []
    for i in range(20, 300, 10):
        r, p, s = get_metrics(minSize=i)
        recall.append(r); precision.append(p); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "min size", recall, precision, None, "min_size_comparison.png")

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
    range_parameter = []
    success = []
    for i in range(30, 300, 10):
        r, p, s = get_metrics(maxSize=i)
        recall.append(r); precision.append(p); range_parameter.append(i/100.0); success.append(s)
    
    plot_metrics(range_parameter, "max size", recall, precision, None, "max_size_comparison.png")

    plt.close()
    plt.plot(range_parameter, success)
    plt.savefig("output/success_maxSize.png")
    plt.close()


if __name__ == "__main__":
    test_scale()
    test_min_size()
    test_min_neigh()
    test_max_size()