import os
import math
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical, register_keras_serializable
from tqdm import tqdm

try:
    from tensorflow.keras.optimizers import AdamW
    HAS_ADAMW = True
except ImportError:
    from tensorflow.keras.optimizers import Adam
    HAS_ADAMW = False

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class CONFIG:
    MODE = 3
    IMG_SIZE = 32
    PATCH_SIZE = 4
    N_SPECTRAL_BANDS = 24
    SPECTRAL_WIDTH = 50
    PROJECTION_DIM = 192
    TRANSFORMER_LAYERS = 6
    NUM_HEADS = 8
    MLP_RATIO = 4
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 1e-3
    LABEL_SMOOTHING = 0.2
    SAVE_DIR = "results"
    CACHE_DIR = "spectral_cache"
    REQUIRE_GPU = True


def configure_runtime():
    print("=" * 60)
    print("TensorFlow version:", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", gpus)

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"CUDA GPU detected: {len(gpus)} physical GPU(s), {len(logical_gpus)} logical GPU(s)")
            print("TensorFlow will use GPU automatically when supported.")
        except RuntimeError as e:
            print("Failed to set GPU memory growth:", e)
    else:
        print("No TensorFlow GPU was detected.")
        print("The program will continue with CPU execution.")

    print("=" * 60)


def add_noise(image, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


def apply_augmentation(images):
    augmented = []

    for img in images:
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        augmented.append(img)

    return np.array(augmented, dtype=np.float32)


def apply_protanopia(rgb_imgs):
    matrix = np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]
    ], dtype=np.float32)

    transformed = np.dot(rgb_imgs, matrix.T) * 0.9 + 0.1

    transformed = np.clip(
        transformed + np.random.normal(0, 0.5, transformed.shape),
        0,
        1
    )

    transformed = transformed * 0.8 + 0.1
    return transformed.astype(np.float32)


def rgb_to_highres_spectral(rgb_img):
    wavelengths = np.linspace(400, 1000, CONFIG.N_SPECTRAL_BANDS)

    curves = [
        np.exp(-(wavelengths - 450) ** 2 / (2 * 30 ** 2)),
        np.exp(-(wavelengths - 550) ** 2 / (2 * 40 ** 2)),
        np.exp(-(wavelengths - 650) ** 2 / (2 * 35 ** 2)),
        0.7 * np.exp(-(wavelengths - 850) ** 2 / (2 * 50 ** 2))
    ]

    spectral_matrix = np.array(curves, dtype=np.float32).T

    rgb_ext = np.concatenate([
        rgb_img,
        np.mean(rgb_img, axis=-1, keepdims=True)
    ], axis=-1).astype(np.float32)

    if rgb_ext.shape[-1] != spectral_matrix.shape[1]:
        raise ValueError(
            f"Input channel mismatch: rgb_ext.shape[-1]={rgb_ext.shape[-1]}, "
            f"spectral_matrix.shape[1]={spectral_matrix.shape[1]}"
        )

    hs = np.einsum("...c,bc->...b", rgb_ext, spectral_matrix).astype(np.float32)

    photon_noise = np.sqrt(np.abs(hs)) * np.random.normal(0, 0.005, hs.shape)
    read_noise = np.random.normal(0, 0.002, hs.shape)
    hs = hs + photon_noise + read_noise

    for idx in [0, 5, 12, 18]:
        hs[..., idx] *= 1.3 + np.random.normal(0, 0.1)

    return hs.astype(np.float32)


def apply_spectral_conversion(images, tag):
    os.makedirs(CONFIG.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CONFIG.CACHE_DIR, f"spectral_{tag}.npy")

    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        spectral_data = np.load(cache_path).astype(np.float32)
    else:
        print(f"Generating new spectral data: {tag}")

        images = np.array(
            [add_noise(img, 0.03) for img in images],
            dtype=np.float32
        )

        hs_images = np.array(
            [
                rgb_to_highres_spectral(img)
                for img in tqdm(images, desc=f"{tag} spectral conversion")
            ],
            dtype=np.float32
        )

        flattened = hs_images.reshape(-1, CONFIG.N_SPECTRAL_BANDS)

        processor = PCA(n_components=12)
        enhanced = processor.fit_transform(flattened)

        scaler = StandardScaler()
        enhanced = scaler.fit_transform(enhanced)

        spectral_data = enhanced.reshape(
            len(images),
            CONFIG.IMG_SIZE,
            CONFIG.IMG_SIZE,
            -1
        ).astype(np.float32)

        np.save(cache_path, spectral_data)

    return spectral_data.astype(np.float32)


@register_keras_serializable(package="Custom")
class PositionEmbedding(layers.Layer):
    def __init__(self, max_length, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, self.max_length, self.embedding_dim),
            initializer="random_normal",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return self.position_embeddings[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "embedding_dim": self.embedding_dim
        })
        return config


class EpochTimeLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"\nStarting epoch {epoch + 1}/{self.params['epochs']}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        print(f"Finished epoch {epoch + 1}/{self.params['epochs']} in {elapsed:.2f} seconds")


def create_optimizer():
    if HAS_ADAMW:
        try:
            return AdamW(
                learning_rate=CONFIG.LEARNING_RATE,
                weight_decay=CONFIG.WEIGHT_DECAY
            )
        except TypeError:
            return AdamW(
                learning_rate=CONFIG.LEARNING_RATE
            )

    return Adam(
        learning_rate=CONFIG.LEARNING_RATE
    )


def create_enhanced_vit():
    input_channels = 12 if CONFIG.MODE == 2 else 3

    inputs = layers.Input(
        shape=(CONFIG.IMG_SIZE, CONFIG.IMG_SIZE, input_channels)
    )

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="swish",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(inputs)

    if CONFIG.MODE != 1:
        x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        CONFIG.PROJECTION_DIM,
        kernel_size=CONFIG.PATCH_SIZE,
        strides=CONFIG.PATCH_SIZE,
        padding="same",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)

    x = layers.Reshape((-1, CONFIG.PROJECTION_DIM))(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    pos_embed = PositionEmbedding(
        max_length=1000,
        embedding_dim=CONFIG.PROJECTION_DIM
    )(x)

    x = x + pos_embed

    for _ in range(CONFIG.TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        attn = layers.MultiHeadAttention(
            num_heads=CONFIG.NUM_HEADS,
            key_dim=CONFIG.PROJECTION_DIM // CONFIG.NUM_HEADS,
            dropout=CONFIG.DROPOUT_RATE,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x1, x1)

        x = layers.Add()([x, attn])

        x2 = layers.LayerNormalization(epsilon=1e-6)(x)

        x2 = layers.Dense(
            CONFIG.PROJECTION_DIM * CONFIG.MLP_RATIO,
            activation="swish",
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x2)

        x2 = layers.Dropout(CONFIG.DROPOUT_RATE)(x2)

        x2 = layers.Dense(
            CONFIG.PROJECTION_DIM,
            kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
        )(x2)

        x = layers.Add()([x, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(
        256,
        activation="swish",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)

    if CONFIG.MODE != 1:
        x = layers.BatchNormalization()(x)

    outputs = layers.Dense(
        10,
        activation="softmax",
        kernel_regularizer=regularizers.l2(CONFIG.WEIGHT_DECAY)
    )(x)

    optimizer = create_optimizer()

    loss_fn = CategoricalCrossentropy(
        from_logits=False,
        label_smoothing=CONFIG.LABEL_SMOOTHING
    )

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    return model


def make_datasets(x_train, y_train):
    steps_per_epoch = math.ceil(len(x_train) / CONFIG.BATCH_SIZE)

    train_ds_fit = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds_fit = (
        train_ds_fit
        .shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
        .batch(CONFIG.BATCH_SIZE)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds_eval = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds_eval = (
        train_ds_eval
        .batch(CONFIG.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds_fit, train_ds_eval, steps_per_epoch


def load_cifar10_batch_from_tar(tar, member_name):
    import pickle

    member = tar.getmember(member_name)
    file_obj = tar.extractfile(member)

    if file_obj is None:
        raise FileNotFoundError(f"Cannot read {member_name} from archive.")

    with file_obj as f:
        batch = pickle.load(f, encoding="latin1")

    data = batch["data"]
    labels = batch["labels"]

    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.array(labels).reshape(-1, 1)

    return data, labels


def load_cifar10_from_tar_gz(tar_gz_path):
    import tarfile

    if not os.path.exists(tar_gz_path):
        raise FileNotFoundError(
            f"CIFAR-10 archive not found: {tar_gz_path}\n"
            "Please put cifar-10-python.tar.gz in the same folder as Train.py."
        )

    print("Using local CIFAR-10 archive:", tar_gz_path)

    x_train_list = []
    y_train_list = []

    with tarfile.open(tar_gz_path, "r:gz") as tar:
        for i in range(1, 6):
            member_name = f"cifar-10-batches-py/data_batch_{i}"
            x_batch, y_batch = load_cifar10_batch_from_tar(tar, member_name)
            x_train_list.append(x_batch)
            y_train_list.append(y_batch)

        x_test, y_test = load_cifar10_batch_from_tar(
            tar,
            "cifar-10-batches-py/test_batch"
        )

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    configure_runtime()

    os.makedirs(CONFIG.SAVE_DIR, exist_ok=True)

    print("Loading CIFAR-10 from local tar.gz file...")

    cifar_archive_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cifar-10-python.tar.gz"
    )

    print("CIFAR-10 archive path:", cifar_archive_path)

    (x_train, y_train), _ = load_cifar10_from_tar_gz(cifar_archive_path)

    x_train = (x_train / 255.0).astype(np.float32)
    y_train = y_train.flatten()

    print("Original train shape:", x_train.shape)

    if CONFIG.MODE == 2:
        x_train = apply_spectral_conversion(
            apply_augmentation(x_train),
            tag="train"
        )
    elif CONFIG.MODE == 3:
        x_train = apply_protanopia(
            apply_augmentation(x_train)
        )
    else:
        x_train = apply_augmentation(x_train)
        x_train = add_noise(x_train, noise_factor=0.03)

    y_train = to_categorical(y_train, num_classes=10)

    train_ds_fit, train_ds_eval, steps_per_epoch = make_datasets(x_train, y_train)

    print("steps_per_epoch:", steps_per_epoch)
    print("CONFIG.EPOCHS:", CONFIG.EPOCHS)

    model = create_enhanced_vit()
    model.summary()

    callbacks = [
        EpochTimeLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CONFIG.SAVE_DIR, "best_model.keras"),
            monitor="accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(CONFIG.SAVE_DIR, "training_log.csv")
        ),
        tf.keras.callbacks.TerminateOnNaN()
    ]

    print("=" * 60)
    print("Training started")
    print(f"Total epochs: {CONFIG.EPOCHS}")
    print("=" * 60)

    history = model.fit(
        train_ds_fit,
        epochs=CONFIG.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    print("=" * 60)
    print("fit() has returned")
    print("Actual trained epochs:", len(history.history["loss"]))
    print("History keys:", list(history.history.keys()))
    print("=" * 60)

    best_model_path = os.path.join(CONFIG.SAVE_DIR, "best_model.keras")

    if os.path.exists(best_model_path):
        print(f"Loading best model: {best_model_path}")
        model = tf.keras.models.load_model(
            best_model_path,
            custom_objects={"PositionEmbedding": PositionEmbedding}
        )
    else:
        print("best_model.keras was not found. Using the final model from the last epoch.")

    final_model_path = os.path.join(CONFIG.SAVE_DIR, "model.keras")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    epochs_ran = len(history.history["accuracy"])

    df = pd.DataFrame({
        "epoch": list(range(1, epochs_ran + 1)),
        "train_accuracy": history.history["accuracy"],
        "train_loss": history.history["loss"]
    })

    history_csv_path = os.path.join(CONFIG.SAVE_DIR, "history.csv")
    df.to_csv(history_csv_path, index=False)
    print(f"History saved to: {history_csv_path}")

    y_train_true = np.argmax(y_train, axis=1)

    print("Predicting train set...")
    y_train_pred = np.argmax(
        model.predict(
            train_ds_eval,
            steps=steps_per_epoch,
            verbose=1
        ),
        axis=1
    )

    y_train_pred = y_train_pred[:len(y_train_true)]

    cm_train = confusion_matrix(y_train_true, y_train_pred)

    cm_train_percentage = (
        cm_train.astype("float32")
        / cm_train.sum(axis=1, keepdims=True)
    )

    train_cm_path = os.path.join(
        CONFIG.SAVE_DIR,
        "confusion_matrix_train_percentage.csv"
    )

    pd.DataFrame(cm_train_percentage).to_csv(
        train_cm_path,
        index=False
    )

    print(f"Train confusion matrix saved to: {train_cm_path}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_accuracy"], label="Train Acc")
    plt.legend()
    plt.title("Training Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.legend()
    plt.title("Training Loss")

    curve_path = os.path.join(CONFIG.SAVE_DIR, "training_curves.png")
    plt.savefig(curve_path)
    plt.close()

    print(f"Training curves saved to: {curve_path}")

    if HAS_ADAMW:
        print("Optimizer: AdamW")
    else:
        print("Current TensorFlow does not support AdamW. Falling back to Adam.")
