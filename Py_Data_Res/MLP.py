import csv
import sys
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import regularizers
from keras.optimizers import Adam

import os
import random
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True


seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

layer_count =  int(sys.argv[1])
neurons = int(sys.argv[2])

train_size = int(sys.argv[3]) #1000
test_size = 10000

x_train_filename = 'x_{}.csv'.format(train_size)
y_train_filename = 'y_{}.csv'.format(train_size)
x_test_filename = 'x_{}.csv'.format(test_size)
y_test_filename = 'y_{}.csv'.format(test_size)
p_test_filename = 'p_MLP_L_{}.csv'.format(train_size)

def create_model(layers):
    model = Sequential()
    if layers == 1:
        model.add(Dense(1, activation='linear', input_shape=(46,)))
        return model

    model.add(Dense(neurons, activation='softplus', input_shape=(46,), kernel_regularizer=regularizer))
    if layers > 2:
        for l in range(layers-2):
            model.add(Dense(neurons//2, activation='softplus', kernel_regularizer=regularizer))
    model.add(Dense(1, activation='linear'))
    return model
    

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

stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,patience=60)

batch_size = 32
epochs = 3000
regularizer = regularizers.l2(l=0)

model = create_model(layer_count)

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3))
# model.summary()

start_train = time.time()    
history = model.fit(x_train, y_train, #validation_split=0.2, 
            epochs=epochs, batch_size=batch_size, verbose=0)#, callbacks=[stop_callback])
end_train = time.time()
train_time = end_train - start_train
print("Train Time: %s seconds." % (train_time))

model.save('my_model_{}_{}_{}.h5'.format(train_size, layer_count, neurons), include_optimizer=False)

# score = model.evaluate(x_test, y_test)
# print(score)

start_pred = time.time()
y_pred = model.predict(x_test)
end_pred = time.time()
pred_time = end_pred - start_pred
print("Prediction Time: %s seconds." % (pred_time))

mse = mean_squared_error(y_test, y_pred)
print("MSE")
print(mse)

with open(p_test_filename, mode='w') as f:
    writer = csv.writer(f, delimiter=',')
    for p in y_pred:
        writer.writerow(p)

# with open('mlp_results.csv', 'a+') as f:
#     f.write("{},{},{},{},{},{},{}\n".format(layer_count, neurons, mse, train_time, pred_time, history.history['loss'][-1], history.history['val_loss'][-1]))

# plt.plot(history.history['loss'], label="trianing")
# plt.plot(history.history['val_loss'], label="validation")
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper left')

# plt.show()