import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import layers
import time
import os
from tensorflow.keras.preprocessing.image import img_to_array


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# IMAGE PROCESSING
def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((128, 128))
    img = img_to_array(img)
    img = img.astype('float32')
    img = (img - 127.5)/127.5
    return img


def image_augment(img):
    flipped = tf.image.flip_left_right(img)
    return flipped


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5) / 255.0)
        plt.axis('off')

    plt.savefig('./training_images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close()


# MODEL
def make_generator():
    model = tf.keras.Sequential()

    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))
    assert model.output_shape == (None, 128, 128, 3)

    return model


def make_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation('sigmoid'))

    return model


# LOSS
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake):
    loss = cross_entropy(tf.ones_like(fake), fake)
    return loss


# TRAINING
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        discrim_loss = discriminator_loss(real_output, fake_output)

    gradient_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discrim = disc_tape.gradient(discrim_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradient_of_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discrim, discriminator.trainable_variables))


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        generate_and_save_images(generator, epoch+1, seed)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f"Time for epoch{epoch+1} is {time.time() - start} sec. Total time is {time.time()-begin}")

    generate_and_save_images(generator, epochs, seed)


BATCH_SIZE = 64
EPOCHS = 400
IMAGE_DIRECTORY = 'images/eyes (copy)/'
num_examples_to_generate = 16
noise_dim = 100
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# CREATING DATASET
train_images = []
for filename in os.listdir(IMAGE_DIRECTORY):
    train_images.append(load_image(os.path.join(IMAGE_DIRECTORY, filename)))

train_images = np.array(train_images)
aug_train_images = np.array([image_augment(img) for img in train_images])
aug_train_images = np.concatenate((train_images, aug_train_images), axis=0)

ds = tf.data.Dataset.from_tensor_slices(aug_train_images).shuffle(5000).batch(BATCH_SIZE)

# TRAINING MODEL
generator = make_generator()
discriminator = make_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

begin = time.time()

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(ds, EPOCHS)
