import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_and_normalize_feed(path, otherStations=2, previousTimes=2):
    print("Reading:", path)
    #data = np.genfromtxt(path, delimiter=' ', dtype= np.float)
    data = np.loadtxt(path, delimiter=' ') # slow but effective
    # do all distances
    # set = np.arange(0,otherStations*2, 2)

    print("Normalizing:", path)

    # calculate smallest angle difference to previous value in wind direction
    set = np.arange(otherStations * 2 + 4, otherStations * 2 + 4 + (1 + otherStations) * 6 * (previousTimes + 1), (1 + otherStations) * 6)
    for i in range(otherStations-1, 0, -1):
        data[:, set + i*6] -= data[:, set + (i-1)*6]

        for j in set:
            # some reason need to make this
            s = data[:, j + i * 6] < 0
            data[:, j + i * 6][s] *= -1

            s = data[:, j + i * 6] > 180
            data[:, j + i*6][s] *= -1
            data[:, j + i*6][s] += 360


    # directions (0-360) (rel location, main wind dir)
    set = np.concatenate((np.arange(1,otherStations*2 + 1, 2), np.arange(otherStations * 2 + 4, otherStations*2 + 4 +  (1 + otherStations)*6*(previousTimes + 1), 6)))
    data[:, set] /= 360


    # air temperature main station
    set = np.arange(otherStations * 2 + 0, otherStations*2 + 0 + (1 + otherStations)*6*(previousTimes + 1), 6)
    scaleVal = (17.94 - 5.04) / 2
    data[:, set] -= 5.04 + scaleVal
    data[:, set] /= scaleVal


    # relative humidity 0->1
    set = np.arange(otherStations * 2 + 1, otherStations * 2 + 1 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    data[:, set] /= 100

    # vapor pressure -1->1
    set = np.arange(otherStations * 2 + 2, otherStations * 2 + 2 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    scaleVal = (1.57 - 0.75) / 2
    data[:, set] -= 0.75 + scaleVal
    data[:, set] /= scaleVal

    # wind speed 0->1
    set = np.arange(otherStations * 2 + 3, otherStations * 2 + 3 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    data[:, set] /= 2.31

    # rain 0 -> 1
    set = np.arange(otherStations * 2 + 5, otherStations * 2 + 5 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    data[:, set] /= 0.58

    return data



def get_train_batch(batch_size):
    n_true = batch_size // 3
    n_false = batch_size - n_true

    n_true_sel = np.random.randint(0, len(train_true), size=n_true)
    n_false_sel = np.random.randint(0, len(train_false), size=n_false)


    x = np.concatenate((
        train_true[n_true_sel], train_false[n_false_sel]
    ))
    y = np.zeros((batch_size, 2))
    y[:n_true, 0] = 1
    y[n_false:, 1] = 1

    return x, y

def get_test_batch(batch_size = None): # send all
    if (batch_size == None):
        return get_train_batch(1000)
    return [], []

def neural_network_model(data):
    # hyper parameters
    n_hidden_1 = 100
    n_hidden_2 = 25
    n_input = 78
    n_output = 2

    h_l_1 = {'weights': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
             'biases': tf.Variable(tf.random_normal([n_hidden_1]))}


    h_l_2 = {'weights': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
             'biases': tf.Variable(tf.random_normal([n_hidden_2]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_hidden_2, n_output])),
             'biases': tf.Variable(tf.random_normal([n_output]))}



    l1 = data @ h_l_1['weights'] + h_l_1['biases']
    l1 = tf.nn.tanh(l1)


    l2 = l1 @ h_l_2['weights'] + h_l_2['biases']
    l2 = tf.nn.tanh(l2)



    return l2 @ output_layer['weights'] + output_layer['biases']


def train_neural_network(x):

    # note subjected to softmax
    prediction = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for epoch in range(num_steps):
            epoch_loss = 0
            for _ in range(int(train_size/batch_size)):
                epoch_x, epoch_y = get_train_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, '\t loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(tf.nn.softmax(y), 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            test_x, test_y = get_test_batch()
            print("Accuracy:", accuracy.eval({x: test_x, y: test_y}))






if __name__ == "__main__":
    print("Loading in data ...")


    bp = "C:\\Users\\raymo\\Desktop\\nn_input\\"
    train_true = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_heavy.csv", otherStations=3, previousTimes=2)
    train_false = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_non_heavy.csv", otherStations=3, previousTimes=2)
    train_size = len(train_true) + len(train_false)
    test_true = train_true #load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_heavy.csv", otherStations=3, previousTimes=2)
    test_false = train_false #load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_non_heavy.csv", otherStations=3, previousTimes=2)
    test_size = len(test_true) + len(test_false)

    ## note training is now testing

    print("Initializing NN")
    # learning_rate = 0.001
    num_steps = 10
    batch_size = 128
    display_step = 100

    x = tf.placeholder(tf.float32, [None, 78])
    y = tf.placeholder(tf.float32)

    print("Starting training")
    train_neural_network(x)