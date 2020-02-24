import csv
import glob

train_path = '/Users/eloymarinciudad/Downloads/termisk_dataset/train'
validation_path = '/Users/eloymarinciudad/Downloads/termisk_dataset/validation'
test_path = '/Users/eloymarinciudad/Downloads/termisk_dataset/test'
split_paths = [train_path, validation_path, test_path]
labels = ['0', '1', '2', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

with open('dataset.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'train', 'label'])  # path= train/0/???.png
    for split_path in split_paths:
        if 'train' in split_path:
            train = 1
        elif 'validation' in split_path:
            train = 0
        else:
            train = 0.5

        for label in labels:
            path = split_path + f'/{label}'
            for filename in glob.glob(path + '/*.png'):
                file_path = filename.split('/Users/eloymarinciudad/Downloads/termisk_dataset')[1]
                writer.writerow([file_path, train, label])
