import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#from sklearn.svm import SVC
#from sklearn.model_selection import train_test_split


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images /255, testing_images /255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:1000]
training_labels = training_labels[:1000]
testing_images = testing_images[:200]
testing_labels = testing_labels[:200]

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))  #resolution 32x32 3 color channels
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(64, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(training_images, training_labels, epochs= 4, validation_data=(testing_images, testing_labels))  #jak często widzi te same dane


img = cv2.imread("horse.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"to najprawdopodobniej {class_names[index]}")
#model.save("image_classifier.model")

#model = models.load_image(image....)    #ładowanie zamiast wyższych linijek kodu
print("Brawo")