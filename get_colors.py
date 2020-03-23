import cv2
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import pyplot as plt


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


ima = get_image('/Users/eloymarinciudad/Downloads/colors_new_original/purple/00000041.jpg')
modified_image = cv2.resize(ima, (600, 400), interpolation=cv2.INTER_AREA)
modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

clf = KMeans(n_clusters=5)
labels = clf.fit_predict(modified_image)

counts = Counter(labels)

center_colors = clf.cluster_centers_
# We get ordered colors by iterating through the keys
ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]


plt.figure(figsize=(8, 6))
plt.pie(counts.values(), labels=hex_colors, colors=hex_colors)
# plt.savefig('prueba_purple.jpg')

print(ordered_colors)