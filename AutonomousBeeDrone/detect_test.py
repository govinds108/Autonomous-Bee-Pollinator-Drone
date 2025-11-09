import sys
import cv2
from utils import findFlower

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    img = cv2.imread(path)
    if img is None:
        print(f"Could not open image: {path}")
        return

    img = cv2.resize(img, (720, 440))
    out, info = findFlower(img)
    print('Detection info (center, area):', info)
    cv2.imshow('Flower Detection Test', out)
    print("Press any key in the image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
