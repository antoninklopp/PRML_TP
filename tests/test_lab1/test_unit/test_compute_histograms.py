#!/usr/bin/env python3

from src.colors_to_probabilities import compute_histograms
from src.info_image import get_boolean_mask, get_all_masks
import glob

path_to_image_folder = "Images/"

class TestComputeHistograms:

    def test_files_exists(self):
        """
        Test that all images from the txt can be accessed
        """
        list_images = []
        print("test")

        # Check that the txt files exist
        assert len(glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt")) > 0

        for f in glob.glob(path_to_image_folder + "FDDB-folds/*ellipseList.txt"):
            with open(f) as file_info:
                name_file = path_to_image_folder + file_info.readline().replace("\n", "") + ".jpg"
                if not name_file:
                    break
                number_faces = int(file_info.readline())
                list_info = []
                for _ in range(number_faces):
                    face = [float(i) for i in file_info.readline().replace("  ", " ").replace("\n", "").split(" ")]
                    list_info.append(face)
                mask = get_boolean_mask(name_file, list_info)
                assert mask is not None

    def test_color_mode(self):
        """
        Test the different color modes
        """
        masks = get_all_masks(5)
        compute_histograms(masks)
        compute_histograms(masks, mode_color='rg')
        compute_histograms(masks, Q=32)
        compute_histograms(masks, mode_color='rg', Q=32)
