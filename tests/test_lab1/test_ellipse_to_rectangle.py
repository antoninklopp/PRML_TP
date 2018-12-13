from src.info_image import ellipse_to_rectangles, get_all_masks
import cv2

if __name__ == "__main__":
    ellipse_to_rectangles()

    with open("Images/rectangle.txt") as f:
        a = f.readline().split(" ")
        name = a[0]
        number = a[1]
        print(a)
        corner_x, corner_y, length = int(a[2]), int(a[3]), int(a[4])
        img = cv2.imread("Images/" + name)
        cv2.rectangle(img, (corner_x, corner_y), (corner_x + length, corner_y + length), (255, 0, 0))
        cv2.imwrite("Images/output.png", img)
