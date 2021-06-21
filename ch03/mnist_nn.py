import sys
import os
import numpy as np
import pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
np.seterr(over='ignore')


def sigmoid(x): return 1.0/(1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


"""Show the first image"""
# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()
#
#
# img, label = x_train[0], t_train[0]
# img = img.reshape(28, 28)
# img_show(img)


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


"""Batched prediction"""
for batch_size in range(10, 101, 10):
    x, t = get_data()
    network = init_network()
    # batch_size = 100
    accuracy_count = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_count += np.sum(p == t[i:i+batch_size])
    print(f'Batch size: {batch_size} Accuracy: {str(float(accuracy_count)/len(x))}')

"""Individual predictions"""
# x, t = get_data()
# network = init_network()
# accuracy_count = 0
# for i in range(0, len(x)):
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     accuracy_count += p == t[i]
# print(f'Accuracy: {str(float(accuracy_count)/len(x))}')
