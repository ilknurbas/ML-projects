import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Conv2DTranspose
from keras.utils import to_categorical


def roc_curve_calc(ground_truth, output):
    print(ground_truth.shape)  # (1000,)
    print(output.shape)  # (1000,)
    print(len(ground_truth))
    print(output)

    # positives
    positives = sum(ground_truth)
    # negatives
    negatives = len(ground_truth) - positives

    # initialize
    fpr = np.zeros(len(output) + 2)
    tpr = np.zeros(len(output) + 2)
    thresholds = sorted(output, reverse=True)  # decrease

    print(len(thresholds))
    print(len(fpr))

    i = 1
    for threshold in thresholds:
        # mark the elements having bigger than the threshold as 1, others 0
        predicted = (output >= threshold).astype(int)

        # True positives
        tp = sum((predicted == 1) & (ground_truth == 1))
        # False Positives
        fp = sum((predicted == 1) & (ground_truth == 0))

        # TPR (True Positive Rate) = # True positives / # positives = Recall = TP / (TP+FN)
        # FPR (False Positive Rate) = # False Positives / # negatives = FP / (FP+TN)
        fpr[i] = fp / negatives
        tpr[i] = tp / positives

        i = i + 1

    fpr[-1] = 1
    tpr[-1] = 1

    return fpr, tpr, thresholds


def part1():
    # roc helps you to choose the best threshold
    ground_truth = np.loadtxt("detector_groundtruth.dat")
    output = np.loadtxt("detector_output.dat")
    ground_truth = 1 - ground_truth

    # using ready-made function
    # true positive rate (recall) vs false positive rate
    fpr, tpr, thresholds = roc_curve(ground_truth, output)
    print(fpr)

    # own implementation
    fpr_1, tpr_1, thresholds_1 = roc_curve_calc(ground_truth, output)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fpr, tpr, color='red')
    ax1.set_xlabel('false positive rate')
    ax1.set_ylabel('true positive rate (recall)')
    ax1.set_title('ready-made function ROC curve')
    ax2.plot(fpr_1, tpr_1, color='blue')
    ax2.set_xlabel('false positive rate')
    ax2.set_ylabel('true positive rate (recall)')
    ax2.set_title('Own implementation ROC curve')
    plt.show()


def part2_cnn(denoised_images):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images/255.0
    test_images = test_images/255.0

    print(train_images.shape)  # (60000, 28, 28)
    print(test_images.shape)  # (10000, 28, 28)

    noise_factor = 0.2
    train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
    test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)
    # Make sure values still in (0,1)
    train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
    test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    train_labels_onehot = to_categorical(train_labels, num_classes=10)
    history = model.fit(train_images, train_labels_onehot, epochs=10, shuffle=True)

    test_labels_onehot = to_categorical(test_labels, num_classes=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels_onehot)
    print('Test accuracy for the clean test images (trained w/clean images):', test_acc)

    test_loss, test_acc = model.evaluate(test_images_noisy, test_labels_onehot)
    print('Test accuracy for the noisy test images (trained w/clean images):', test_acc)

    test_loss, test_acc = model.evaluate(denoised_images, test_labels_onehot)
    print('Test accuracy for denoised noisy test images (trained w/clean images): ', test_acc)

    # train with noisy # COMPARE WITH AUTOENCODER
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    train_labels_onehot = to_categorical(train_labels, num_classes=10)
    history = model.fit(train_images_noisy, train_labels_onehot, epochs=10, shuffle=True)

    test_labels_onehot = to_categorical(test_labels, num_classes=10)
    test_loss, test_acc = model.evaluate(test_images_noisy, test_labels_onehot)
    print('Test accuracy for the noisy test images (trained w/noisy images):', test_acc)


def part2_autoencoder():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print(train_images.shape)  # (60000, 28, 28)
    print(test_images.shape)  # (10000, 28, 28)

    # added
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    # added

    noise_factor = 0.2
    train_images_noisy = train_images + noise_factor * tf.random.normal(shape=train_images.shape)
    test_images_noisy = test_images + noise_factor * tf.random.normal(shape=test_images.shape)
    # Make sure values still in (0,1)
    train_images_noisy = tf.clip_by_value(train_images_noisy, clip_value_min=0., clip_value_max=1.)
    test_images_noisy = tf.clip_by_value(test_images_noisy, clip_value_min=0., clip_value_max=1.)

    model2 = Sequential([
        # encoder
        Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', strides=2, input_shape=(28, 28, 1)),
        Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),

        # decoder
        Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')

    ])

    model2.summary()
    model2.compile(loss='mse', optimizer='adam')

    history = model2.fit(train_images_noisy, train_images, epochs=5, shuffle=True,
                         validation_data=(test_images_noisy, test_images))

    # PRINT THE OUTPUT AND COMPARE
    denoised_images = model2.predict(test_images_noisy)

    plt.figure(figsize=(6, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images_noisy[i])
        plt.axis("off")
    plt.show()

    plt.figure(figsize=(6, 5))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(denoised_images[i])
        plt.axis("off")
    plt.show()

    return denoised_images


def main_start():
    ## PART 1
    part1()  # uncomment

    ## PART 2 autoencoder
    denoised_images = part2_autoencoder()  # uncomment

    ## PART 2 cnn
    part2_cnn(denoised_images)  # uncomment


if __name__ == '__main__':
    main_start()
