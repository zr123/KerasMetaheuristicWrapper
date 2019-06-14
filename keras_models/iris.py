from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from keras_models.DummyOptimizer import DummyOptimizer


def get_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    opt = DummyOptimizer()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def get_data():
    data = load_iris()
    x_train = data.data
    # one-hot encode (categorical encode) y_train
    encoder = LabelEncoder()
    encoder.fit(data.target)
    encoded_y = encoder.transform(data.target)
    y_train = np_utils.to_categorical(encoded_y)
    return x_train, y_train
