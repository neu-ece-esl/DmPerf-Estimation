import csv
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

train_size = 5000
test_size = 10000

x_train_filename = 'x_{}.csv'.format(train_size)
y_train_filename = 'y_{}.csv'.format(train_size)
x_test_filename = 'x_{}.csv'.format(test_size)
y_test_filename = 'y_{}.csv'.format(test_size)
p_test_filename = 'p_Linear_{}.csv'.format(train_size)

def open_dataset(x_filename, y_filename):
    x_train = []
    y_train = []
    with open(x_filename, 'r') as xf, open(y_filename, 'r') as yf:
        xreader = csv.reader(xf, delimiter=',')
        yreader = csv.reader(yf, delimiter=',')
        for x, y in zip(xreader, yreader):
            x_train.append(x)
            y_train.append(y)
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
    return x_train, y_train

x_train, y_train = open_dataset(x_train_filename, y_train_filename)
print("Loaded training set, size: {}".format(x_train.shape))
x_test, y_test = open_dataset(x_test_filename, y_test_filename)
print("Loaded test set, size: {}".format(x_test.shape))
#print("x_train")
#print(x_train)
#print("y_train")
#print(y_train)


start_train = time.time()
reg = LinearRegression().fit(x_train, y_train)
end_train = time.time()
print("Train Time: %s seconds." % (end_train - start_train))

start_pred = time.time()
y_pred = reg.predict(x_test)
end_pred = time.time()
print("Prediction Time: %s seconds." % (end_pred - start_pred))

#print("y_pred")
#print(y_pred)

with open(p_test_filename, mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for p in y_pred:
        writer.writerow(p)

mse = mean_squared_error(y_test, y_pred)
print("MSE")
print(mse)