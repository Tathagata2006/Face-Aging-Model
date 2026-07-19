import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Hyperparameters & Configuration
# -------------------------------------------------------------
INPUT_DIM = 784       # 28x28 flattened images
HIDDEN_DIM = 128      # Dimension of hidden layers
LATENT_DIM = 10       # Dimension of latent space z
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# -------------------------------------------------------------
# 2. Model Architecture Definitions
# -------------------------------------------------------------

def build_encoder():
    model = tf.keras.Sequential([
        layers.Input(shape=(INPUT_DIM,)),
        layers.Dense(HIDDEN_DIM, activation='relu'),
        layers.Dense(LATENT_DIM) # Latent code z (Linear activation)
    ], name="Encoder")
    return model

def build_decoder():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(HIDDEN_DIM, activation='relu'),
        layers.Dense(INPUT_DIM, activation='sigmoid') # Normalized outputs [0, 1]
    ], name="Decoder")
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(HIDDEN_DIM, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Probability of being from true prior
    ], name="Discriminator")
    return model

# Initialize components
encoder = build_encoder()
decoder = build_decoder()
discriminator = build_discriminator()

# Optimizers
ae_optimizer = optimizers.Adam(learning_rate=LR)
dc_optimizer = optimizers.Adam(learning_rate=LR)
gen_optimizer = optimizers.Adam(learning_rate=LR)

# Loss Function Helper
cross_entropy = losses.BinaryCrossentropy()

# -------------------------------------------------------------
# 3. Custom Training Step (The Core AAE Logic)
# -------------------------------------------------------------

@tf.function
def train_step(images):
    # --- PHASE 1: Reconstruction (Autoencoder) ---
    with tf.GradientTape() as ae_tape:
        latent_z = encoder(images, training=True)
        reconstructed_images = decoder(latent_z, training=True)
        
        # Binary Cross Entropy for image reconstruction
        recon_loss = tf.reduce_mean(losses.binary_crossentropy(images, reconstructed_images))
        
    ae_gradients = ae_tape.gradient(recon_loss, encoder.trainable_variables + decoder.trainable_variables)
    ae_optimizer.apply_gradients(zip(ae_gradients, encoder.trainable_variables + decoder.trainable_variables))

    # --- PHASE 2: Discriminator Update ---
    # Sample real noise from the chosen prior distribution (Gaussian)
    true_prior_z = tf.random.normal(shape=(tf.shape(images)[0], LATENT_DIM), mean=0.0, stddev=1.0)
    
    with tf.GradientTape() as dc_tape:
        latent_z = encoder(images, training=False)
        
        d_real_output = discriminator(true_prior_z, training=True)
        d_fake_output = discriminator(latent_z, training=True)
        
        # Discriminator wants to label prior as 1, encoder output as 0
        dc_loss_real = cross_entropy(tf.ones_like(d_real_output), d_real_output)
        dc_loss_fake = cross_entropy(tf.zeros_like(d_fake_output), d_fake_output)
        dc_loss = dc_loss_real + dc_loss_fake
        
    dc_gradients = dc_tape.gradient(dc_loss, discriminator.trainable_variables)
    dc_optimizer.apply_gradients(zip(dc_gradients, discriminator.trainable_variables))

    # --- PHASE 3: Generator (Encoder) Update ---
    with tf.GradientTape() as gen_tape:
        latent_z = encoder(images, training=True)
        d_fake_output = discriminator(latent_z, training=True)
        
        # Encoder wants Discriminator to think its latent space output is real (1)
        gen_loss = cross_entropy(tf.ones_like(d_fake_output), d_fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, encoder.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_gradients, encoder.trainable_variables))
    
    return recon_loss, dc_loss, gen_loss

# -------------------------------------------------------------
# 4. Data Preparation & Training Loop
# -------------------------------------------------------------

# Load Fashion MNIST (or standard MNIST)
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((-1, INPUT_DIM)) # Flatten to 784

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

print("Starting training loop...")
for epoch in range(EPOCHS):
    recon_epoch_loss = 0
    dc_epoch_loss = 0
    gen_epoch_loss = 0
    num_batches = 0
    
    for image_batch in train_dataset:
        r_loss, d_loss, g_loss = train_step(image_batch)
        recon_epoch_loss += r_loss
        dc_epoch_loss += d_loss
        gen_epoch_loss += g_loss
        num_batches += 1
        
    print(f"Epoch {epoch+1:02d}/{EPOCHS:02d} | "
          f"Recon Loss: {recon_epoch_loss/num_batches:.4f} | "
          f"Disc Loss: {dc_epoch_loss/num_batches:.4f} | "
          f"Gen Loss: {gen_epoch_loss/num_batches:.4f}")

# -------------------------------------------------------------
# 5. Image Generation (Testing Phase)
# -------------------------------------------------------------
print("\nGenerating synthetic images from Gaussian prior noise...")

# Sample pure noise vectors directly from our Gaussian distribution
random_noise = tf.random.normal(shape=(10, LATENT_DIM), mean=0.0, stddev=1.0)
generated_images = decoder(random_noise, training=False).numpy()

# Plot the generated items
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.suptitle("Generated Images from Latent Space Prior")
plt.show()