import math as m

def dist(a, b):
    """
    Distance entre les vecteurs a et b
    """
    return m.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1])** 2)

def overlapping(true_rectangle, predicted_rectangle):
    """
    Vérifie si deux rectangles s'overlappent
    """

    middle_rec1 = (true_rectangle[0] + true_rectangle[2]/2.0, true_rectangle[1] + true_rectangle[3]/2.0)
    middle_rec2 = (predicted_rectangle[0] + predicted_rectangle[2]/2.0, predicted_rectangle[1] + predicted_rectangle[3]/2.0)

    # On prend la moitié du premier rectangle comme threshold pour l'overlapping
    if dist(middle_rec1, middle_rec2) > true_rectangle[2]/2: 
        return -1 # La valeur de retour -1 indique que les deux rectangles ne s'overlappent pas. 
    
    # On trouve le rectangle correspondant à la coordonnée
    # maximale du coin gauche dans les deux directions
    max_x_l = max(true_rectangle[0], predicted_rectangle[0]) # max x, l (left corner)
    max_y_l = max(true_rectangle[1], predicted_rectangle[1]) # max y, l (left corner)

    min_x_r = min(true_rectangle[0] + true_rectangle[2], predicted_rectangle[0] + predicted_rectangle[2]) # max x, l (left corner)
    min_y_r = max(true_rectangle[1] + true_rectangle[3], predicted_rectangle[1] + predicted_rectangle[3]) # max y, l (left corner)

    # On trouve l'overlapping
    # Cela correpond à la plus grande coordonnées en des coins gauche 
    # et la plus petite des coins en bas à droite
    overlapping = (min_x_r - max_x_l) * (min_y_r - max_y_l)

    # On regarde maintenant le rapport entre l'overlapping et les rectangles
    surface_true = true_rectangle[2] * true_rectangle[3]

    # print("true rec", true_rectangle)
    # print("predicted rec", predicted_rectangle)
    # print("overlap", overlapping/surface_true)

    return overlapping/surface_true


def overlapping_predicted(predicted_rectangle, list_true, threshold):
    """
    Return if a predicted rectangle is overlapping enough a true one
    """
    overlap = -1
    for t in list_true:
        new_o = overlapping(t, predicted_rectangle)
        if new_o > overlap:
            overlap = new_o

    if overlap > threshold:
        return True
    else:
        return False

