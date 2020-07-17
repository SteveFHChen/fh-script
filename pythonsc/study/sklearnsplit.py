import numpy as np
from sklearn.model_selection import train_test_split

x, y = np.arange(10).reshape((5, 2)), range(5)

x
y
list(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train, x_test, y_train, y_test

