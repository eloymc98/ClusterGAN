import argparse
import os
import cv2


def resize_google_colors(size):
    path = "/content/ClusterGAN/colors/google_colors"
    dirs = os.listdir(path)
    print(dirs)
    for item in dirs:
        subdir = path + '/' + item
        print(subdir)
        if os.path.isdir(subdir):
            for image in os.listdir(subdir):
                print(image)
                if os.path.isfile(image):
                    bgr = cv2.imread(image)
                    # img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(subdir + '/' + image, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='colors')
    parser.add_argument('--size', type=int, default=32)

    args = parser.parse_args()

    if args.data == 'colors':
        resize_google_colors(args.size)
