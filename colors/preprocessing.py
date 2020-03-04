import csv
import glob
import random

data_path = '/Users/eloymarinciudad/Downloads/google_colors'
split_paths = [data_path]
labels = ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

with open('dataset.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['path', 'train', 'label'])  # path= train/0/???.png
    for split_path in split_paths:
        for label in labels:
            path = split_path + f'/{label}'
            for filename in glob.glob(path + '/*.jpg'):
                train = 1 if random.uniform(0, 1) >= 0.2 else 0
                file_path = filename.split('/Users/eloymarinciudad/Downloads/google_colors')[1]
                writer.writerow([file_path, train, label])
