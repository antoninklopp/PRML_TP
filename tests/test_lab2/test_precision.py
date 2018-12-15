from src.lab2.lab2 import *
from src.metrics.metrics import get_all_metric, plot_metrics

def get_metrics(numImg=50, scale=2, minNeigh=5, minSize=30, maxSize=200):
    """
    Testing the metrics by changing parameters

    All the parameters are the one used in the get_true_predicted_faces function. 

    :return: recall, precision, accuracy
    """
    infos_file_path = ROOT_PATH+"Images/WIDER/WIDER_train_faces.txt"

    true_masks, predicted_masks, scores = get_true_predicted_faces(infos_file_path, numImg, scale, minNeigh)
    true_masks_flatten = np.concatenate([i.flatten() for i in true_masks])
    predicted_masks_flatten = np.concatenate([i.flatten() for i in predicted_masks])

    print(true_masks_flatten.shape)
    print(predicted_masks_flatten.shape)

    metrics = get_all_metric(true_masks_flatten, predicted_masks_flatten)

    return metrics["recall"], metrics["precision"], metrics["accuracy"]

def test_scale():
    """
    Test the impact of the scale on metrics
    """
    recall = []
    precision = []
    accuracy = []
    range_parameter = []
    for i in range(110, 300, 1):
        r, p, a = get_metrics(scale=i/100.0)
        recall.append(r); precision.append(p); accuracy.append(a); range_parameter.append(i/100.0)
    
    plot_metrics(range_parameter, "scale", recall, precision, accuracy, "scale_comparison.png")


if __name__ == "__main__":
    test_scale()