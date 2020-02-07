import numpy as np
label = np.random.randint(low = 0 , high = 10, size = 50)
print(label)
#print(np.eye(10)[label])

print(np.hstack((0.10 * np.random.randn(50, 10-9),np.eye(10)[label])))