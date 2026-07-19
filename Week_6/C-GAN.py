import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
EPOCHS = 30
BATCH_SIZE = 128
NOISE_DIM = 100
NUM_CLASSES = 10
IMAGE_SHAPE = (28, 28, 1)

# Target resolution for flattening/reshaping
IMAGE_SIZE = 28 * 28 * 1 

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# Loss function (Binary Cross-Entropy)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Fashion MNIST Label Map for visualization reference
LABEL_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==========================================
# 2. Data Preprocessing Pipeline
# ==========================================
print("Loading and preparing Fashion MNIST dataset...")
(train_images, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()

# Normalize images to the range [-1, 1] (Standard practice for GANs)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Convert labels to one-hot vectors
train_labels = tf.one_hot(train_labels, depth=NUM_CLASSES)

# Build the tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(60000).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# ==========================================
# 3. Model Definitions
# ==========================================

def build_generator():
    """
    Inputs: 
      - Noise vector: (BATCH_SIZE, NOISE_DIM)
      - Class label vector: (BATCH_SIZE, NUM_CLASSES)
    Output: 
      - Generated flat image: (BATCH_SIZE, IMAGE_SIZE)
    """
    # Define functional API inputs
    noise_input = layers.Input(shape=(NOISE_DIM,), name="noise_input")
    label_input = layers.Input(shape=(NUM_CLASSES,), name="label_input")
    
    # Structural Concatenation of random features and class rules
    x = layers.Concatenate()([noise_input, label_input])
    
    # Dense hidden layers
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    
    # Output layer maps to flattened image size using tanh (-1 to 1)
    output_layer = layers.Dense(IMAGE_SIZE, activation='tanh')(x)
    
    model = models.Model(inputs=[noise_input, label_input], outputs=output_layer, name="Generator")
    return model


def build_discriminator():
    """
    Inputs:
      - Flat image vector: (BATCH_SIZE, IMAGE_SIZE)
      - Class label vector: (BATCH_SIZE, NUM_CLASSES)
    Output:
      - Scalar validity prediction: (BATCH_SIZE, 1)
    """
    image_input = layers.Input(shape=(IMAGE_SIZE,), name="image_input")
    label_input = layers.Input(shape=(NUM_CLASSES,), name="label_input")
    
    # Structural Concatenation of target image and standard layout rules
    x = layers.Concatenate()([image_input, label_input])
    
    # Dense hidden layers
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Single-node raw output (logits) for validation estimation
    output_layer = layers.Dense(1)(x)
    
    model = models.Model(inputs=[image_input, label_input], outputs=output_layer, name="Discriminator")
    return model

# Instantiate networks
generator = build_generator()
discriminator = build_discriminator()


# ==========================================
# 4. Custom Training Loop Logic
# ==========================================

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(real_images, labels):
    # Sample random noise vectors
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    
    # Flatten real images to match network input shape
    real_images_flat = tf.reshape(real_images, [BATCH_SIZE, IMAGE_SIZE])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 1. Generate synthetic images using the noise and conditional labels
        fake_images_flat = generator([noise, labels], training=True)
        
        # 2. Evaluate discriminator on real samples vs fake samples
        real_output = discriminator([real_images_flat, labels], training=True)
        fake_output = discriminator([fake_images_flat, labels], training=True)
        
        # 3. Calculate separate adversarial losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    # 4. Calculate gradients and optimize weights
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss


def train(dataset, epochs):
    print("Beginning Training Loop...")
    for epoch in range(epochs):
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        
        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(image_batch, label_batch)
            gen_loss_avg.update_state(g_loss)
            disc_loss_avg.update_state(d_loss)
            
        print(f"Epoch {epoch+1}/{epochs} | Gen Loss: {gen_loss_avg.result():.4f} | Disc Loss: {disc_loss_avg.result():.4f}")
        
        # Periodically show sample visual feedback during execution
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generate_and_plot_samples(epoch + 1)


# ==========================================
# 5. Targeted Conditional Generation Routine
# ==========================================

def generate_and_plot_samples(epoch_num=None, target_class_idx=9):
    """
    Generates a 4x4 grid targeting a specific clothing item index.
    Default index 9 represents 'Ankle boot'.
    """
    test_noise = tf.random.normal([16, NOISE_DIM])
    
    # Create matching labels mapping exclusively to the targeted index
    test_labels = tf.one_hot(tf.repeat(target_class_idx, 16), depth=NUM_CLASSES)
    
    # Generate images from inputs
    generated_flat = generator([test_noise, test_labels], training=False)
    
    # Reshape vectors back to visual image matrices
    generated_images = tf.reshape(generated_flat, (-1, 28, 28))
    
    fig, axes = plt.subplots(4, 4, figsize=(4, 4))
    for i, ax in enumerate(axes.flat):
        # Scale back to display range [0, 1] from [-1, 1]
        ax.imshow(generated_images[i] * 0.5 + 0.5, cmap='gray')
        ax.axis('off')
        
    title_suffix = f" at Epoch {epoch_num}" if epoch_num else ""
    plt.suptitle(f"Generated Category: {LABEL_NAMES[target_class_idx]}{title_suffix}", fontsize=12)
    plt.show()


# ==========================================
# 6. Execution Block
# ==========================================
if __name__ == "__main__":
    # Run the main training sequence
    train(dataset, EPOCHS)
    
    # Test generation for alternate target labels post-training
    print("\nTraining completed! Testing custom target class generations...")
    generate_and_plot_samples(epoch_num=None, target_class_idx=4)  # Generate 'Coat'
    generate_and_plot_samples(epoch_num=None, target_class_idx=1)  # Generate 'Trouser'