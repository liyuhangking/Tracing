import cv2
import numpy as np

def nothing(x):
    pass

def process_image(image, threshold_low, threshold_high):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, threshold_low, threshold_high)
    return binary

def main():
    img = cv2.imread('2.png')
    cv2.namedWindow('image')

    # 设置窗口大小
    cv2.resizeWindow('image', 800, 600)

    cv2.createTrackbar('Hue_Low', 'image', 0, 255, nothing)
    cv2.createTrackbar('Saturation_Low', 'image', 0, 255, nothing)
    cv2.createTrackbar('Value_Low', 'image', 0, 255, nothing)
    cv2.createTrackbar('Hue_High', 'image', 255, 255, nothing)
    cv2.createTrackbar('Saturation_High', 'image', 255, 255, nothing)
    cv2.createTrackbar('Value_High', 'image', 255, 255, nothing)

    while True:
        h_low = cv2.getTrackbarPos('Hue_Low', 'image')
        s_low = cv2.getTrackbarPos('Saturation_Low', 'image')
        v_low = cv2.getTrackbarPos('Value_Low', 'image')
        h_high = cv2.getTrackbarPos('Hue_High', 'image')
        s_high = cv2.getTrackbarPos('Saturation_High', 'image')
        v_high = cv2.getTrackbarPos('Value_High', 'image')

        threshold_low = np.array([h_low, s_low, v_low])
        threshold_high = np.array([h_high, s_high, v_high])

        result = process_image(img, threshold_low, threshold_high)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        cv2.imshow('image', np.hstack([img, result]))

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
