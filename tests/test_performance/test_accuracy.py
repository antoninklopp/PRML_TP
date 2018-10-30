from src.colors_to_probabilities import *
from src.info_image import *

class TestClass:

    def test_accuracy(self):
        seuil = 0.7
        res_t, res_th = load_histograms(number_files=10, recompute=True)
        test_files = get_mask_from_file("Images/FDDB-folds/FDDB-fold-10-ellipseList.txt", 50)
        prediction_precision = []
        for name, image in test_files:
            image_test = cv2.imread(name)
            prediction = get_prediction(image_test, res_t, res_th, seuil)
            print(np.sum(prediction))
            prediction_precision.append(np.sum(prediction[np.where(image==1)])/np.sum(prediction))
        print(prediction_precision)

t = TestClass()
t.test_accuracy()