import tensorflow as tf
import numpy as np
# from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets, layers, models,preprocessing
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns; 
sns.set()



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model2 = models.load_model('kazem.h')
# print(test_images.shape)

predictions=model2.predict(test_images)
print(predictions)
predictions = np.argmax(predictions, axis = 1)
# for i in predictions:
#     print(class_names[i])



# plt.figure(figsize=(10, 10))
# for i in range(10):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i])
#     # The CIFAR labels happen to be arrays,
#     # which is why you need the extra index
#     plt.xlabel(class_names[test_labels[i][0]])
# plt.show()




# cm = confusion_matrix(test_labels, predictions)

# plt.figure(figsize=(9,9))
# sns.heatmap(cm, cbar=False, xticklabels=class_names, yticklabels=class_names, fmt='d', annot=True, cmap=plt.cm.Blues)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


img = keras.preprocessing.image.load_img("frog.jpg", target_size=(32, 32))
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
image_tensor = np.vstack([x])
classes = model2.predict(image_tensor / 255.0)
predictions = np.argmax(classes, axis = 1)[0]
print(predictions)
print(class_names[predictions])