import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import product

IMAGE_SIZE = 28
LABELS = 10


def read_data():
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    return train_data, test_data


def find_images(n_cols: int = 4) -> np.ndarray:
    """read the data from train.csv and return 4 different examples from each class"""
    image_count = 0
    data_to_visualize = np.zeros((LABELS * n_cols, IMAGE_SIZE ** 2))
    sort_train_data = train_data.sort_values(by='label')
    for i in range(LABELS):
        data_to_visualize[image_count:image_count + n_cols, :] = sort_train_data[sort_train_data['label'] == i].iloc[
                                                                 :n_cols, 1:]
        image_count += n_cols
    return data_to_visualize


def visualize_data(n_cols: int = 4):
    """plot 4 different examples from each class in a
    grid of 10x4. Label each row by the class name"""
    data_to_visualize = find_images(n_cols)
    fig, axes = plt.subplots(10, n_cols, figsize=(10, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = data_to_visualize[i, :].reshape(IMAGE_SIZE, IMAGE_SIZE)
        if i % n_cols == 0:
            ax.set_ylabel(true_labels[i // n_cols], rotation='horizontal', fontsize=12, labelpad=20.0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.imshow(img, cmap='gray')
    # plt.show()


def normalize(x: pd.DataFrame) -> pd.DataFrame:
    """normalize the data using min-max normalization"""
    eps = 1e-8  # add small epsilon for numerical stability
    return (x - x.min()) / (x.max() - x.min() + eps)


def arrange_data() -> np.ndarray:
    """arrange the data in the correct format for the classifier"""
    train_data_labels = train_data['label']
    train_data_without_labels = train_data.drop('label', axis=1)
    # Step 1 - Split the train data to train part and validation part
    val_pct = 0.2  # set percentage for validation split
    random_state = 42  # random seed for reproducibility
    x_train, x_val, y_train, y_val = train_test_split(train_data_without_labels, train_data_labels, test_size=val_pct,
                                                      random_state=random_state, stratify=train_data_labels,
                                                      shuffle=True)

    # Step 2 - normalizes the data
    x_train = normalize(x_train)
    x_val = normalize(x_val)
    x_test = normalize(test_data)

    # Step 3 - one-hot encode the labels
    y_train_one_hot = np.zeros((y_train.size, LABELS))
    y_train_one_hot[np.arange(y_train.size), y_train] = 1
    y_val_one_hot = np.zeros((y_val.size, LABELS))
    y_val_one_hot[np.arange(y_val.size), y_val] = 1

    # Step 4 - add constant "1" for the bias term
    x_train = np.column_stack((x_train, np.ones(x_train.shape[0])))
    x_val = np.column_stack((x_val, np.ones(x_val.shape[0])))
    x_test = np.column_stack((x_test, np.ones(x_test.shape[0])))

    return x_train, y_train_one_hot, x_val, y_val_one_hot, x_test


# helper functions for the logistic regression classifier
def softmax(z: np.ndarray) -> np.ndarray:
    numerator = np.exp(z - np.max(z, axis=0))
    denominator = np.sum(np.exp(z - np.max(z, axis=0)), axis=0)
    return numerator / denominator


def cross_entropy(softmax_z: np.ndarray, y_train_one_hot: np.ndarray) -> np.ndarray:
    n = softmax_z.shape[1]
    loss = np.sum(np.log(softmax_z + 10e-8) * y_train_one_hot.T)
    return -loss / n


def derive_loss(softmax_z: np.ndarray, y_train_one_hot: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
    n = softmax_z.shape[1]
    dL = (softmax_z - y_train_one_hot).T @ x_batch
    return dL / n


class LogisticRegression:
    """implement the logistic regression classifier"""

    def __init__(self, batch_size, learning_rate, regularization_coefficient):
        self.bs = batch_size
        self.lr = learning_rate
        self.rc = regularization_coefficient
        self.train_loss, self.train_accuracy, self.validation_loss, self.validation_accuracy = ([] for _ in range(4))

    def train(self, epochs: int = 64):
        n_samples = x_train.shape[0]  # number of training samples
        n_features = x_train.shape[1]

        w = np.random.normal(loc=0.0, scale=0.01, size=(LABELS, n_features))

        for _ in range(epochs):  # iterations over entire dataset
            loss = 0
            accuracy = 0
            for batch_idx, idx_start in enumerate(
                    range(0, n_samples, self.bs)):  # batch iterations whithin each dataset iteration (epoch)
                idx_end = min(idx_start + self.bs, n_samples)
                x_batch = x_train[idx_start:idx_end, :]  # take all data in the current batch
                y_batch_onehot = y_train_one_hot[idx_start:idx_end, :]  # take all labels in the current batch
                z = w @ x_batch.T
                # calc. probaility of y_j = 1 for each input (M,)
                softmax_of_z = softmax(z)
                # calculate loss and accuracy
                loss += cross_entropy(softmax_of_z, y_batch_onehot) + self.rc * np.sum(np.power(w, 2))
                predicts = np.argmax(softmax_of_z, axis=0)
                accuracy += np.mean(np.equal(predicts, np.argmax(y_batch_onehot, axis=1)))
                # compute gradient of the loss w.r.t W
                gradients = derive_loss(softmax_of_z.T, y_batch_onehot, x_batch) + 2 * self.rc * w
                # update W
                w -= self.lr * gradients

            validation_acc_epoch, validation_loss_epoch = self._validate(w)  ##### validation #####

            self.train_accuracy.append(accuracy / (batch_idx + 1))
            self.train_loss.append(loss / (batch_idx + 1))
            self.validation_accuracy.append(validation_acc_epoch)
            self.validation_loss.append(validation_loss_epoch)

        self._plot(epochs)
        return w

    def test(self, w: np.ndarray) -> None:
        z = w @ x_test.T
        softmax_of_z = softmax(z)
        predicts = np.argmax(softmax_of_z, axis=0)
        np.savetxt('lr_pred.csv', predicts, fmt='%i')

    def _validate(self, w: np.ndarray) -> tuple[np.ndarray]:
        z = w @ x_val.T
        softmax_of_z = softmax(z)
        predicts = np.argmax(softmax_of_z, axis=0)
        loss = cross_entropy(softmax_of_z, y_val_one_hot) + self.rc * np.sum(np.power(w, 2))
        accuracy = np.mean(np.equal(predicts, np.argmax(y_val_one_hot, axis=1)))
        return accuracy, loss

    def _plot(self, epochs: int) -> None:
        steps = np.arange(epochs)
        fig, axs1 = plt.subplots()

        axs1.set_xlabel('epochs')
        axs1.set_ylabel('Accuracy')
        axs1.set_title(
            'Batch size: ' + str(self.bs) + ', Learning rate: ' + str(self.lr) + ', Regularization coefficient '
            + str(self.rc) + '\nTrain accuracy: %.3f, Train loss: %.3f\n'
                             'Validation accuracy: %.3f, Validation loss: %.3f' % (self.train_accuracy[-1],
                                                                                   self.train_loss[-1],
                                                                                   self.validation_accuracy[-1],
                                                                                   self.validation_loss[-1]))
        axs1.plot(steps, self.train_accuracy, label="train accuracy", color='blue')
        axs1.plot(steps, self.validation_accuracy, label="validation accuracy", color='black')
        axs2 = axs1.twinx()
        axs2.set_ylabel('Loss')
        axs2.plot(steps, self.train_loss, label="train loss", color='red')
        axs2.plot(steps, self.validation_loss, label="validation loss", color='green')
        fig.legend(loc="upper right")

        print('\nBatch size: ' + str(self.bs) + ', Learning rate: ' + str(self.lr) + ', Regularization coefficient '
              + str(self.rc) +
              '\n Train accuracy: %.5f, Train loss: %.5f\n Validation accuracy: %.5f, Validation loss: %.5f' %
              (self.train_accuracy[-1], self.train_loss[-1], self.validation_accuracy[-1], self.validation_loss[-1]))


# different activation functions for the neural network
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def derive_sigmoid(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def derive_relu(z: np.ndarray) -> np.ndarray:
    derive = np.zeros(z.shape)
    derive[z > 0] = 1
    return derive


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def derive_tanh(z: np.ndarray) -> np.ndarray:
    return 1 - np.power(tanh(z), 2)


class NeuralNetwork:
    """implement the neural network classifier with one hidden layer"""

    def __init__(self, batch_size, learning_rate, regularization_coefficient, hidden_size, activation_function,
                 dropout):
        self.bs = batch_size
        self.lr = learning_rate
        self.rc = regularization_coefficient
        self.hs = hidden_size
        self.af = activation_function
        self.do = dropout
        self.train_loss, self.train_accuracy, self.validation_loss, self.validation_accuracy = ([] for _ in range(4))

    def train(self, epochs: int = 64, iteration: int = 0):

        n_samples = x_train.shape[0]  # number of training samples
        n_features = x_train.shape[1]
        w1 = np.random.normal(loc=0.0, scale=0.01, size=(self.hs, n_features))
        b1 = np.zeros((self.hs, 1))
        w2 = np.random.normal(loc=0.0, scale=0.01, size=(self.hs, LABELS))
        b2 = np.zeros((LABELS, 1))

        for _ in tqdm(range(epochs), colour="green", desc="Iteration "+str(iteration)+"\t"):  # iterations over entire dataset
            loss = 0
            accuracy = 0
            for batch_idx, idx_start in enumerate(
                    range(0, n_samples, self.bs)):  # batch iterations whithin each dataset iteration (epoch)
                idx_end = min(idx_start + self.bs, n_samples)
                x_batch = x_train[idx_start:idx_end, :]  # take all data in the current batch
                y_batch_onehot = y_train_one_hot[idx_start:idx_end, :]  # take all labels in the current batch
                z1 = w1 @ x_batch.T + b1
                h = self.af(z1)
                y_hat = softmax(w2.T @ h + b2)

                # calculate loss and accuracy
                loss += cross_entropy(y_hat, y_batch_onehot) + self.rc * (
                        np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))
                predicts = np.argmax(y_hat, axis=0)
                accuracy += np.mean(np.equal(predicts, np.argmax(y_batch_onehot, axis=1)))

                if self.do != 1:
                    # apply dropout - keep unit with probability of do (dropout)
                    drop_idx = np.random.randint(0, h.shape[0], int(self.hs * (1 - self.do)))
                    mask = np.ones((h.shape[0], 1))
                    mask[drop_idx, 0] = 0
                    h = h * mask

                # derive gradients using the quotient rule of derivatives
                diff = y_hat.T - y_batch_onehot
                dLdb2 = (1 / n_samples) * np.sum(diff, axis=0, keepdims=True)
                dLdw2 = (1 / n_samples) * h @ diff

                dLdh = diff @ w2.T
                if self.af == relu:
                    dLdz1 = dLdh * derive_relu(z1).T
                if self.af == sigmoid:
                    dLdz1 = dLdh * derive_sigmoid(z1).T
                if self.af == tanh:
                    dLdz1 = dLdh * derive_tanh(z1).T

                dLdw1 = (1 / n_samples) * (dLdz1.T @ x_batch)
                dLdb1 = (1 / n_samples) * np.sum(dLdz1, axis=0, keepdims=True)

                # gradient descent
                w2 -= self.lr * (dLdw2 + 2 * self.rc * w2)
                b2 -= self.lr * dLdb2.T
                w1 -= self.lr * (dLdw1 + 2 * self.rc * w1)
                b1 -= self.lr * dLdb1.T

            validation_acc_epoch, validation_loss_epoch = self._validate(w1, w2, b1, b2)  ##### validation #####

            self.train_accuracy.append(accuracy / (batch_idx + 1))
            self.train_loss.append(loss / (batch_idx + 1))
            self.validation_accuracy.append(validation_acc_epoch)
            self.validation_loss.append(validation_loss_epoch)

        self._print_results()

        if self.do != 1:
            w1, w2, b1, b2 = self.do * w1, self.do * w2, self.do * b1, self.do * b2

        return w1, b1, w2, b2

    def _validate(self, w1, w2, b1, b2) -> tuple[np.ndarray, np.ndarray]:
        h = self.af(w1 @ x_val.T + b1)
        y_hat = softmax(w2.T @ h + b2)
        y_pred = np.argmax(y_hat, axis=0)
        loss = cross_entropy(y_hat, y_val_one_hot) + self.rc * (
                np.sum(np.power(w1, 2)) + np.sum(np.power(w2, 2)))
        accuracy = np.mean(np.equal(y_pred, np.argmax(y_val_one_hot, axis=1)))
        return accuracy, loss

    def test(self, w1, b1, w2, b2) -> None:
        h = self.af(w1 @ x_test.T + b1)
        y_hat = softmax(w2.T @ h + b2)
        y_pred = np.argmax(y_hat, axis=0)
        np.savetxt('NN_pred.csv', y_pred, fmt='%i')

    def _print_results(self) -> None:
        print('\nBatch size: ' + str(self.bs) + '\tLearning rate: ' + str(
            self.lr) + '\tRegularization coefficient: ' + str(self.rc)
              + '\tActivation Function: ' + str(self.af.__name__) + '\tHidden Size: ' + str(
            self.hs) + '\tDropout Prob: ' + str(self.do) +
              '\n Train accuracy: %.5f, Train loss: %.5f\n Validation accuracy: %.5f, Validation loss: %.5f' %
              (self.train_accuracy[-1], self.train_loss[-1], self.validation_accuracy[-1], self.validation_loss[-1]))

    def plot(self, epochs: int, va, vl, ta, tl, bs, lr, rc, af, hs, do) -> None:
        steps = np.arange(epochs)
        fig, axs1 = plt.subplots()

        axs1.set_xlabel('epochs')
        axs1.set_ylabel('Accuracy')
        axs1.set_title(
            'Batch size: ' + str(bs) + ', Learning rate: ' + str(lr) + ', Regularization coefficient: '
            + str(rc) + ', Activation function: ' + af + ', Hidden size: ' + str(hs) + ', Dropout prob(keep): ' + str(
                do) +
            '\n\nTrain accuracy: %.5f, Train loss: %.5f, Validation accuracy: %.5f, Validation loss: %.5f' % (
                ta[-1], tl[-1],
                va[-1], vl[-1]))
        axs1.plot(steps, ta, label="train accuracy", color='blue')
        axs1.plot(steps, va, label="validation accuracy", color='black')
        axs2 = axs1.twinx()
        axs2.set_ylabel('Loss')
        axs2.plot(steps, tl, label="train loss", color='red')
        axs2.plot(steps, vl, label="validation loss", color='green')
        fig.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    train_data, test_data = read_data()
    true_labels = ["T-Shirt",  # index 0
                   "Trouser",  # index 1
                   "Pullover",  # index 2
                   "Dress",  # index 3
                   "Coat",  # index 4
                   "Sandal",  # index 5
                   "Shirt",  # index 6
                   "Sneaker",  # index 7
                   "Bag",  # index 8
                   "Ankle boot"]  # index 9

    # Part 1 - Visualize the data
    visualize_data(4)

    # Part 2 - Logistic Regression Classifier
    x_train, y_train_one_hot, x_val, y_val_one_hot, x_test = arrange_data()

    # hyper-parameter search
    batch_sizes = [128, 256, 512]
    learning_rates = [0.001, 0.01, 0.05]
    regularization_coefficients = [10e-8, 10e-4]

    w, val_acc = [], []
    total_iterations = len(batch_sizes) * len(learning_rates) * len(regularization_coefficients)

    for batch_size, learning_rate, regularization_coefficient in tqdm(
            product(batch_sizes, learning_rates, regularization_coefficients), total=total_iterations):
        Lr = LogisticRegression(batch_size, learning_rate, regularization_coefficient)
        w.append(Lr.train(epochs=500))
        val_acc.append(Lr.validation_accuracy[-1])  # save the validation accuracy of the last epoch

    plt.show()
    # test and save the results of the best logistic regression classifier
    LR = LogisticRegression(batch_size=0, learning_rate=0,
                            regularization_coefficient=0)  # create an instance of the class
    # Params here doesn't matter because we already trained the classifiers
    LR.test(w[np.argmax(val_acc)])  # test the best classifier
    print("saved the results of the best logistic regression classifier to lr_pred.csv")

    # Part 3 - Neural Network with One Hidden Layer

    # drop the bias added in the previous part
    x_train = x_train[:, :-1]
    x_val = x_val[:, :-1]
    x_test = x_test[:, :-1]

    # hyper-parameter search
    batch_sizes = [128]
    learning_rates = [0.5]
    regularization_coefficients = [1e-08]
    activation_functions = [relu, sigmoid, tanh]
    h_sizes = [256, 128, 10]
    dropout_prob = [1, 0.9, 0.8, 0.5]

    max_val_acc = 0
    w1, b1, w2, b2, val_acc, val_loss, train_acc, train_loss = [], [], [], [], [], [], [], []
    bs, lr, rc, hs, do = 0, 0, 0, 0, 0
    af = ''

    i = 0

    for batch_size, learning_rate, regularization_coefficient, activation_function, h_size, drop in product(
            batch_sizes, learning_rates, regularization_coefficients, activation_functions, h_sizes, dropout_prob):

        NN = NeuralNetwork(batch_size, learning_rate, regularization_coefficient, h_size, activation_function, drop)
        w1_i, b1_i, w2_i, b2_i = NN.train(epochs=300, iteration=i)
        i += 1

        if NN.validation_accuracy[-1] > max_val_acc:
            # save the best classifier Params for plotting
            max_val_acc = NN.validation_accuracy[-1]
            val_acc = NN.validation_accuracy
            val_loss = NN.validation_loss
            train_acc = NN.train_accuracy
            train_loss = NN.train_loss
            bs = batch_size
            lr = learning_rate
            rc = regularization_coefficient
            af = activation_function.__name__
            hs = h_size
            do = drop
            # save the best classifier weights for testing
            w1 = w1_i
            b1 = b1_i
            w2 = w2_i
            b2 = b2_i

    # params of the best classifier (just the af relevant for the test function)
    NN = NeuralNetwork(bs, lr, rc, hs, af, do)

    # plot results of the best neural network classifier
    NN.plot(epochs=300, va=val_acc, vl=val_loss, ta=train_acc, tl=train_loss, bs=bs, lr=lr, rc=rc, af=af,
            hs=hs, do=do)

    NN.test(w1, b1, w2, b2)
    print("saved the results of the best logistic regression classifier to NN_pred.csv")
