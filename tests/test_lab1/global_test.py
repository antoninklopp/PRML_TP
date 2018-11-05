from src.info_image import get_all_masks, get_test_masks, get_training_masks
from src.colors_to_probabilities import get_prediction, load_histograms

def _test():

    # Train model 
    masks_training = get_training_masks()
    h, hT = load_histograms(recompute=True, masks=masks_training)

    # Evaluate model
    masks_test = get_test_masks()