# CNN for mel frequency spectrogram 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

phonemes = ["t", "th", "tt"]

labels = []
train_images = []

train_dir = 'trainspecs/'
train_files = os.listdir(train_dir)

for png in train_files:
    file_info = png.split("-")
    start_idx = 1 if len(file_info[0]) < 5 else 2
    end_idx = 1 if len(file_info[0]) == 3 else 2 if len(file_info[0]) < 6 else 3
    phoneme = file_info[0][start_idx:end_idx+1]
    label = phonemes.index(phoneme)
    labels.append(label)

    img = cv2.imread(os.path.join(train_dir, png))
    train_images.append(img)

test_images = []

test_dir = 'testspecs/'
test_files = os.listdir(test_dir)

for png in test_files:
    img = cv2.imread(os.path.join(test_dir, png))
    test_images.append(img)
    
train_images = np.array(train_images)
labels = pd.get_dummies(labels).values
# labels = np.array(labels)
test_images = np.array(test_images)
    
x_train, x_test, y_train, y_test = train_test_split(train_images, labels, test_size = 0.2, random_state=1)

num_neurons = 100
batch_size = 30
epochs = 15

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(217, 334, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

model.save('spec-cnn-model')

predict = np.squeeze(model.predict(test_images))
pred_phonemes = [phonemes[np.argmax(lab)] for lab in predict]

# output to csv
pred_frame = pd.DataFrame(columns=['filename', 'label'])
filenames = [file[:-4] + ".wav" for file in test_files]
pred_frame = pd.DataFrame({'filename': filenames, 'label': pred_phonemes})

pred_frame.to_csv('img-cnn.csv', encoding='utf-8', index=False)

# %% Heatmap generation
import tensorflow as tf
import tensorflow.keras.backend as K
import keras
import numpy as np
import os
import cv2

model = keras.models.load_model('spec-cnn-model')

# orig_img = 'testspecs/utu-12_testing.png'
orig_img = 'testspecs/uttu-18_testing.png'
# orig_img = 'testspecs/uthu-6_testing.png'

img = np.array([cv2.imread(orig_img)])
preds = model.predict(img)

with tf.GradientTape() as tape:
  last_conv_layer = model.get_layer('conv2d_35')
  iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
  model_out, last_conv_layer = iterate(img)
  class_out = model_out[:, np.argmax(model_out[0])]
  grads = tape.gradient(class_out, last_conv_layer)
  pooled_grads = K.mean(grads, axis=(0, 1, 2))
  
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap = heatmap.reshape((50, 80))
plt.matshow(heatmap)
plt.show()

pic = cv2.imread(orig_img)
intensity = 0.5
heatmap = cv2.resize(heatmap, (pic.shape[1], pic.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
pic = heatmap * intensity + pic
plt.imshow(pic)