import numpy as np
import cv2

count = 0
for i in range(11):
    r = i * 0.1
    for j in range(11):
        g = j * 0.1
        for k in range(11):
            b = k * 0.1
            ima = np.array([ [[r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b], [r,g,b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]],
                             [[r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b],
                              [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b], [r, g, b]]])
            rgb = ima[:, :, [2, 1, 0]] * 255
            cv2.imwrite(f'synthetic_colors/{round(r,1)}_{round(g,1)}_{round(b,1)}_colors{count}.jpg', rgb)
            count += 1
            if count % 100 == 0:
                print(count)

