import numpy as np
import pickle
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import time

CATEGORIES = ["Dog", "Cat"]

with open("PetData.pickle", "rb") as d:
    data = pickle.load(d)
    d.close()

with open("PetLabels.pickle", "rb") as l:
    labels = pickle.load(l)
    l.close()

DATA_LENGTH = len(data)
data = np.array(data/255.0).reshape(-1, 50, 50, 1)
labels = np.array(labels)

training_data, test_data = data[0:math.floor(DATA_LENGTH*.9)],  \
                           data[math.ceil(DATA_LENGTH*.9): -1]
training_labels, test_labels = labels[0:math.floor(DATA_LENGTH*.9)], \
                               labels[math.ceil(DATA_LENGTH*.9): -1]

model = tf.keras.Sequential()

model.add(Conv2D(64, (3, 3), input_shape=training_data.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("sigmoid"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

print(model)

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(training_data, training_labels, batch_size=128, validation_split=0.1, epochs=10)

time_stamp= time.strftime("%Y%m%d-%H%M%S")
model.save(f"PetNetwork-{time_stamp}.h5")
# for i in range(10):
#     plt.imshow(test_data[i], cmap=plt.cm.binary)
#     plt.xlabel(f"Actual: {test_labels[i]}")
#     plt.title(f"Prediction: {model.predict(test_data[i])}")
#     plt.show()

# print(len(training_data), len(training_labels), len(test_labels), len(test_labels))

