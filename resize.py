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
                if os.path.isfile(subdir + '/' + image):
                    print(image)
                    bgr = cv2.imread(subdir + '/' + image)
                    # img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(subdir + '/' + image, img)


def resize_termisk(size):
    path = "/content/ClusterGAN/termisk/termisk_dataset"
    dirs = os.listdir(path)
    print(dirs)
    for item in dirs:
        subdir = path + '/' + item
        print(subdir)
        if os.path.isdir(subdir):
            dirs2 = os.listdir(subdir)
            for item2 in dirs2:
                subdir2 = subdir + '/' + item2
                if os.path.isdir(subdir2):
                    for image in os.listdir(subdir2):
                        if os.path.isfile(subdir2 + '/' + image):
                            print(image)
                            grey = cv2.imread(subdir2 + '/' + image, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(subdir2 + '/' + image, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='colors')
    parser.add_argument('--size', type=int, default=32)

    args = parser.parse_args()

    if args.data == 'colors':
        resize_google_colors(args.size)
    elif args.data == 'termisk':
        resize_termisk(args.size)