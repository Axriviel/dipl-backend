import tarfile
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Rozbalení CIFAR-10 archivu
def extract_cifar10(file_path, extract_to):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

# Načtení CIFAR-10 binárních dat
def load_cifar10(data_folder):
    def unpickle(file):
        with open(file, 'rb') as f:
            return pickle.load(f, encoding='bytes')

    train_images, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(f"{data_folder}/data_batch_{i}")
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])

    train_images = np.vstack(train_images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    test_batch = unpickle(f"{data_folder}/test_batch")
    test_images = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch[b'labels'])

    return (train_images, train_labels), (test_images, test_labels)

# Připrava CIFAR-10
cifar10_path = "cifar-10-python.tar.gz"
extract_to = "cifar-10"
extract_cifar10(cifar10_path, extract_to)

data_folder = f"{extract_to}/cifar-10-batches-py"
(train_images, train_labels), (test_images, test_labels) = load_cifar10(data_folder)

# Rozdělení dat na trénovací a validační sady
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

x_train=train_images 
y_train=train_labels
x_val=val_images 
y_val=val_labels
x_test=test_images 
y_test=test_labels

from tensorflow.keras.utils import to_categorical

# Normalizace obrázků na rozsah [0, 1]
x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding štítků
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Uložení normalizovaných dat
np.savez_compressed("cifar10_normalized.npz",
         x_train=x_train, y_train=y_train,
         x_val=x_val, y_val=y_val,
         x_test=x_test, y_test=y_test)

print("Data byla připravena a uložena jako cifar10_prepared.npz")
