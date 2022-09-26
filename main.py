import tensorflow as tf
import matplotlib.pyplot as plt

img_height, img_width = 32, 32
batch_size = 20

# this fetch the batch of images from the path indicated and setting the image size and width and batch size
# 70% for training
train_ds = tf.keras.utils.image_dataset_from_directory(
    "leafs/train",
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# 10% for validating
val_ds = tf.keras.utils.image_dataset_from_directory(
    "leafs/validation",
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# 20% for testing
test_ds = tf.keras.utils.image_dataset_from_directory(
    "leafs/test",
    image_size = (img_height, img_width),
    batch_size = batch_size
)

# Change this to your desire classification names
class_names = ["basale", "betel", "guava", "jackfruit", "jasmine", "lemon", "mango", "mint", "oleander", "pomegranate", "sandalwood", "tulsi"]
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

