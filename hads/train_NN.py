import tensorflow as tf
from tensorflow import keras
import numpy as np
from hads.training import *
from tensorflow.keras import layers
import pandas as pd

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

def X_split(X,y):
    X = np.asarray(X)
    y = np.asarray(y)
    X = normalized(X)
    y = standardized(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    print('X_train.shape={}\n y_train.shape={}\n X_test.shape={}\n y_test.shape={}\n'.format(X_train.shape,
                                                                                             y_train.shape,
                                                                                             X_test.shape,
                                                                                             y_test.shape))
    return X_train, X_test, y_train, y_test


def train_nn(X_caled,y_caled):
    X_train, X_test, y_train, y_test = X_split(X_caled,y_caled)
    EPOCHS = 2000
    model = build_model(X_train.shape[1])
    print(model.summary())
    # history = model.fit(
    #     X_train, y_train,
    #     epochs=EPOCHS, validation_split=0.2, verbose=2,
    #     callbacks=[PrintDot()])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_split=0.2, verbose=10, callbacks=[early_stop, PrintDot()])

    plot_history(history)


    test_predictions = model.predict(X_test).flatten()
    # train_predictions = model.predict(X_train).flatten()
    # predictions = model.predict(X_tocal).flatten()
    # print(test_predictions.shape)
    # print(y_test.shape)
    # plt.scatter(y_test, test_predictions)
    # plt.xlabel('True Values [MPG]')
    # plt.ylabel('Predictions [MPG]')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlim([0, plt.xlim()[1]])
    # plt.ylim([0, plt.ylim()[1]])
    # _ = plt.plot([-20, 20], [-20, 20])
    # plt.show()
    #
    # plt.scatter(y_train, train_predictions)
    # plt.xlabel('True Values [MPG]')
    # plt.ylabel('Predictions [MPG]')
    # plt.axis('equal')
    # plt.axis('square')
    # plt.xlim([0, plt.xlim()[1]])
    # plt.ylim([0, plt.ylim()[1]])
    # _ = plt.plot([-20, 20], [-20, 20])
    # plt.show()

    # error = test_predictions - y_test
    # plt.hist(error, bins=25)
    # plt.xlabel("Prediction Error [MPG]")
    # _ = plt.ylabel("Count")
    # plt.show()
    return predictions, test_predictions

def build_model(s):
    model = keras.Sequential([
        layers.Dense(8, activation='relu', input_shape=[s]),
        # layers.Dense(24, activation='relu'),
        layers.Dense(8, activation='relu'),
        # layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    return model


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  # plt.ylim([0,5])
  plt.legend()
  plt.show()

  # plt.figure()
  # plt.xlabel('Epoch')
  # plt.ylabel('Mean Square Error [$MPG^2$]')
  # plt.plot(hist['epoch'], hist['mse'],
  #          label='Train Error')
  # plt.plot(hist['epoch'], hist['val_mse'],
  #          label = 'Val Error')
  # # plt.ylim([0,20])
  # plt.legend()
  # plt.show()

def predict(model, X):
    return model.predict(X)