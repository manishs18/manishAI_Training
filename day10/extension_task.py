""" Extension Task: Generate Higher-Resolution Images with GANs

The basic GAN architecture often generates low-resolution images (e.g., 28×28 for handwritten digits). To generate higher-resolution images (64×64, 128×128, or larger), the network architecture must be enhanced.

Approach 1: Add More Upsampling Layers

Increase the Generator's output size gradually using additional transposed convolution (deconvolution) layers.

Example

Current Generator:

Noise (100)
    ↓
Dense Layer
    ↓
7×7×128
    ↓
ConvTranspose
    ↓
14×14×64
    ↓
ConvTranspose
    ↓
28×28×1

Extended Generator:

Noise (100)
    ↓
Dense Layer
    ↓
4×4×512
    ↓
ConvTranspose
    ↓
8×8×256
    ↓
ConvTranspose
    ↓
16×16×128
    ↓
ConvTranspose
    ↓
32×32×64
    ↓
ConvTranspose
    ↓
64×64×3

This allows the Generator to create 64×64 RGB images.

Approach 2: Progressive Growing GAN (ProGAN)

Instead of training directly at high resolution:

Start training at 4×4.
Expand to 8×8.
Expand to 16×16.
Expand to 32×32.
Expand to 64×64 and beyond.
4×4
 ↓
8×8
 ↓
16×16
 ↓
32×32
 ↓
64×64
 ↓
128×128

Benefits:

More stable training.
Better image quality.
Faster convergence.
Approach 3: Replace ConvTranspose with Upsampling + Convolution

Transposed convolutions sometimes create checkerboard artifacts.

Instead:

x = UpSampling2D(size=(2,2))(x)
x = Conv2D(128, kernel_size=3, padding="same")(x)

Architecture:

Feature Map
     ↓
Upsampling
     ↓
Convolution
     ↓
BatchNorm
     ↓
ReLU

Benefits:

Smoother images.
Fewer artifacts.
Better visual quality.
Improved Discriminator

As image resolution increases, the Discriminator should also become deeper.

Example:

64×64×3
     ↓
Conv2D 64
     ↓
Conv2D 128
     ↓
Conv2D 256
     ↓
Conv2D 512
     ↓
Dense
     ↓
Real/Fake

This enables the model to capture finer image details.

Additional Enhancements
Batch Normalization

Helps stabilize training:

BatchNormalization()
Leaky ReLU

Prevents dead neurons:

LeakyReLU(alpha=0.2)
Wasserstein GAN (WGAN)

Improves training stability for high-resolution generation.

Spectral Normalization

Controls exploding gradients in the Discriminator.

Evaluation Metrics

For higher-resolution GANs, use:

Metric	Purpose
FID (Fréchet Inception Distance)	Measures image quality and realism
IS (Inception Score)	Measures diversity and quality
Precision & Recall for GANs	Measures realism vs diversity
Human Evaluation	Visual quality assessment
Expected Outcome

By adding additional upsampling layers or using Progressive Growing GANs, the model can generate:

28×28  → 64×64 → 128×128 → 256×256

with significantly improved image detail, realism, and visual quality while maintaining stable training.

"""





import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Reshape,
    Conv2D,
    UpSampling2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    Dropout,
    Input
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np

# ==========================================
# CONFIG
# ==========================================

LATENT_DIM = 100
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

# ==========================================
# GENERATOR (64x64 RGB)
# ==========================================

def build_generator():

    model = Sequential(name="Generator")

    # Noise -> 4x4x512
    model.add(Dense(4 * 4 * 512, input_dim=LATENT_DIM))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((4, 4, 512)))

    # 4x4 -> 8x8
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 8x8 -> 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 16x16 -> 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # 32x32 -> 64x64
    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    # Final RGB image
    model.add(
        Conv2D(
            CHANNELS,
            kernel_size=3,
            activation="tanh",
            padding="same"
        )
    )

    return model


# ==========================================
# DISCRIMINATOR
# ==========================================

def build_discriminator():

    model = Sequential(name="Discriminator")

    model.add(
        Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding="same",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        )
    )
    model.add(LeakyReLU(0.2))

    model.add(
        Conv2D(
            128,
            kernel_size=3,
            strides=2,
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(
        Conv2D(
            256,
            kernel_size=3,
            strides=2,
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(
        Conv2D(
            512,
            kernel_size=3,
            strides=2,
            padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))

    return model


# ==========================================
# BUILD MODELS
# ==========================================

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    optimizer=Adam(0.0002, 0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# GAN MODEL
discriminator.trainable = False

noise = Input(shape=(LATENT_DIM,))
generated_image = generator(noise)
validity = discriminator(generated_image)

gan = Model(noise, validity)

gan.compile(
    optimizer=Adam(0.0002, 0.5),
    loss="binary_crossentropy"
)

# ==========================================
# TRAINING LOOP
# ==========================================

def train(real_images, epochs=10000, batch_size=32):

    real_images = real_images.astype(np.float32)
    real_images = (real_images - 127.5) / 127.5

    half_batch = batch_size // 2

    for epoch in range(epochs):

        # -------------------
        # Train Discriminator
        # -------------------

        idx = np.random.randint(
            0,
            real_images.shape[0],
            half_batch
        )

        real_batch = real_images[idx]

        noise = np.random.normal(
            0,
            1,
            (half_batch, LATENT_DIM)
        )

        fake_batch = generator.predict(
            noise,
            verbose=0
        )

        d_loss_real = discriminator.train_on_batch(
            real_batch,
            np.ones((half_batch, 1))
        )

        d_loss_fake = discriminator.train_on_batch(
            fake_batch,
            np.zeros((half_batch, 1))
        )

        d_loss = 0.5 * np.add(
            d_loss_real,
            d_loss_fake
        )

        # -------------------
        # Train Generator
        # -------------------

        noise = np.random.normal(
            0,
            1,
            (batch_size, LATENT_DIM)
        )

        valid_labels = np.ones(
            (batch_size, 1)
        )

        g_loss = gan.train_on_batch(
            noise,
            valid_labels
        )

        # -------------------
        # Logging
        # -------------------

        if epoch % 100 == 0:

            print(
                f"Epoch {epoch} "
                f"D Loss={d_loss[0]:.4f} "
                f"D Acc={d_loss[1]*100:.2f}% "
                f"G Loss={g_loss:.4f}"
            )


# ==========================================
# GENERATE IMAGES
# ==========================================

def generate_images(num_images=5):

    noise = np.random.normal(
        0,
        1,
        (num_images, LATENT_DIM)
    )

    images = generator.predict(
        noise,
        verbose=0
    )

    images = (images + 1) / 2

    return images


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print(generator.summary())
    print(discriminator.summary())

    print("\nGAN ready for training.")
    print("Expected output size:")
    print("(64, 64, 3)")