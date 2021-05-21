import csv
import time
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.model_selection import KFold

train_size = 1000
test_size = 10000

x_train_filename = 'x_{}.csv'.format(train_size)
y_train_filename = 'y_{}.csv'.format(train_size)
x_test_filename = 'x_{}.csv'.format(test_size)
y_test_filename = 'y_{}.csv'.format(test_size)
p_test_filename = 'p_MLP_L_{}.csv'.format(train_size)

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



kfolds = 10
batch_size = 1
epochs = 50
#neurons = [6]
neurons = range(1,50)

means = []
best_model = None
best_score = 10000

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
kfold.get_n_splits(x_train)

start_train = time.time()

for count in neurons:
    cvscores = []
    print("Using {} neurons..".format(count))

    model = Sequential()
    model.add(Dense(count, activation='softplus', input_shape=(46,)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error',
                optimizer='adam')

    weights = model.get_weights()

    for train, valid in kfold.split(x_train, y_train):
        x_train_fold = x_train[train]
        y_train_fold = y_train[train]
        x_valid_fold = x_train[valid]
        y_valid_fold = y_train[valid]
        
        model.fit(x_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)
        score = model.evaluate(x_valid_fold, y_valid_fold, verbose=0)
        print(score)
        cvscores.append(score)
        model.set_weights(weights)

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
    means.append(np.mean(cvscores))
    if score < best_score:
        best_model = model
        best_model.fit(x_valid_fold, y_valid_fold, epochs=epochs, batch_size=batch_size, verbose=0)
        best_score = score

end_train = time.time()
print("Train Time: %s seconds." % (end_train - start_train))

print(means)
print("Best Number of Neurons")
print(means.index(min(means)))

score = best_model.evaluate(x_test, y_test)
print(score)

start_pred = time.time()
y_pred = best_model.predict(x_test)
end_pred = time.time()
print("Prediction Time: %s seconds." % (end_pred - start_pred))


with open(p_test_filename, mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for p in y_pred:
        writer.writerow(p)

mse = mean_squared_error(y_test, y_pred)
print("MSE")
print(mse)
