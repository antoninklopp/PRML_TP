#!/usr/bin/env python3

from src.colors_to_probabilities import *

class TestComputeHistograms:

    def test_create(self):
        """
        A method to test the creation and the load of all histograms
        """
        compute_histograms(number_files=100)

    def test_files_exists(self):
        """
        Test that all images from the txt can be accessed
        """
        list_images = []
        print("test")
        for f in glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"):
            with open(f) as file_info:
                while (image_max >= 0) or (all is True):
                    name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
                    if not name_file:
                        break
                    number_faces = int(file_info.readline())
                    list_info = []
                    for _ in range(number_faces):
                        face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                        list_info.append(face)
                    mask = get_boolean_mask(name_file, list_info)
                    if mask is not None:
                        list_images.append([name_file, mask])
                    else:
                        return False
        return True