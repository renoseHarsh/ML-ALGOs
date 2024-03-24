from AllAlgos import MultiLinearRegression
import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
model = MultiLinearRegression(X_train, y_train, 0.000000001, 2000)

model.fit()
print(model.predict(np.array([2104, 5, 1, 45])))