import keras
import keras.layers as layers
from keras import Model, Sequential
import keras.models
from keras import optimizers, applications
import tensorflow as tf
import keras.backend as K


def f1_score(self, y_pred):
    true_positives = K.sum(K.round(K.clip(self * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(self, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*(precision*recall)/(precision+recall+K.epsilon())


base_model = applications.ResNet50(weights='imagenet',
                                           include_top=False,
                                           input_shape=(48, 48, 3))

for layer in base_model.layers[:-4]:
    layer.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation="relu"))
    model.add(keras.layers.Dense(7, activation="softmax"))
    model.add(keras.layers.Dropout(0.5))

input_shape = (None, 48, 48, 3)
model.build(input_shape)

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    f1_score,
]

model.compile(optimizers.Adam(learning_rate=0.0001,
                              decay=1e-4), loss='MSE', metrics=METRICS)
model.summary()
model.save('mymodel.h5')
