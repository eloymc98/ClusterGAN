# import numpy as np
#
# label = np.random.randint(low=0, high=10, size=50)
# print(label)
# # print(np.eye(10)[label])
#
# print(np.hstack((0.10 * np.random.randn(50, 10 - 9), np.eye(10)[label])))

# from pathlib import Path
#
# ima = Path('/Users/eloymarinciudad/Downloads/')
#
# if ima.is_file():
#     print('File')
# if ima.is_dir():
#     print('Dir')

import cv2
import numpy as np
img = cv2.imread('/Users/eloymarinciudad/Downloads/300.0.png', cv2.IMREAD_GRAYSCALE)
res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
res = res.flatten()
latent = np.zeros(shape=(1, 40))
x = np.zeros(shape=(1, res.shape[0]))
x[0, :] = res

print(x[0])
