import src.metrics as met
import numpy as np

MACHINE_ENSIMAG = False
try:
    import matplotlib.pyplot as plt
except:
    MACHINE_ENSIMAG = True

class TestPlot:

    def ROC(self):
        """
        entraine et trace la roc
        """
        pass

    def Precision_Recall(self):
        """
        entraine et trace la courbe Precision = f(recall)
        :return:
        """
        pass

if __name__ == "__main__":
    print("nothing to do yet")
