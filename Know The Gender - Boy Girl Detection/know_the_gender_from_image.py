import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop

training = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

# Chargement des images d'entraînement et de validation
training_dataset = training.flow_from_directory("training_images", target_size=(200, 200), batch_size=3, class_mode="binary")
validation_dataset = validation.flow_from_directory("validation_images", target_size=(200, 200), batch_size=3, class_mode="binary")

print(training_dataset.class_indices)
print(validation_dataset.classes)

# Construction du model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ]
)

# Compilation du model
model.compile(optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Entraînement du model
train = model_fit = model.fit(training_dataset, epochs=40, validation_data=validation_dataset)

# Graphe perte par itération
plt.title("Taux de parte par itération")
plt.plot(train.history['loss'], label="perte")
plt.plot(train.history['val_loss'], label="valeur perte")
plt.legend()
plt.show()

# Graphe précision par itération
plt.title("Taux de précision par itération")
plt.plot(train.history['accuracy'], label="précision")
plt.plot(train.history['val_accuracy'], label="valeur précision")
plt.legend()
plt.show()

# Exemple d'aperçu de la classification avec des tests
dir_path = "testing_images"

for i in os.listdir(dir_path):
    img_path = dir_path + "/" + i
    img = image.load_img(img_path, target_size=(200, 200))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    val = model.predict(images)
    print(val)

    if val == 0:
        print("C'est un garçon")
        plt.title("BOY")
    else:
        print("C'est une fille")
        plt.title("GIRL")

    plt.imshow(img)
    plt.show()
