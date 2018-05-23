import tensorflow as tf
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def load_and_normalize_feed(path, otherStations=2, previousTimes=2, nrow=None):
    if (nrow != None):
        n = sum(1 for line in open(path))  # number of records in file
        print("Selecting only", np.round(nrow * 100 / n, decimals=4), '% of', path)
        nrow = sorted(np.random.choice(n, n-nrow, replace=False))
    else:
        nrow = 0

    print("Reading:", path)
    data = np.array(read_csv(path, delimiter=' ', skiprows=nrow, header=None))
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
    n_true = int(batch_size / (train_size/len(train_true)))
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
        batch_size = test_size
    n_true = int(batch_size / (test_size/len(test_true)))
    n_false = batch_size - n_true

    n_true_sel = np.random.randint(0, len(test_true), size=n_true)
    n_false_sel = np.random.randint(0, len(test_false), size=n_false)

    x = np.concatenate((
        test_true[n_true_sel], test_false[n_false_sel]
    ))
    y = np.zeros((batch_size, 2))
    y[:n_true, 0] = 1
    y[n_false:, 1] = 1

    return x, y


def neural_network_model(data):
    # hyper parameters
    n_hidden_1 = 120
    n_hidden_2 = 20
    n_input = 78
    n_output = 2

    print("Variables:", (n_input+1)*n_hidden_1 + (n_hidden_1+1)*n_hidden_2 + (n_hidden_2 + 1)*n_output)

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


    s_prediction = tf.nn.softmax(prediction)

    false_positive = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(s_prediction, 1)-tf.argmax(y, 1), 1), tf.float32))

    correct = tf.equal(tf.argmax(s_prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        for epoch in range(n_epoch):
            epoch_loss = 0
            for _ in range(int(train_size/batch_size)):
                epoch_x, epoch_y = get_train_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            epoch_loss /= int(train_size/batch_size)

            train_loss_plot[0].append(epoch)
            train_loss_plot[1].append(epoch_loss)
            if (epoch % display_step == 0):
                test_x, test_y = get_test_batch()
                t_loss, acc, f_p =  sess.run([cost, accuracy, false_positive], feed_dict={x: test_x, y: test_y})
                test_loss_plot[0].append(epoch)
                test_loss_plot[1].append(t_loss)
                test_acc_plot[0].append(epoch)
                test_acc_plot[1].append(acc)
                print('Epoch', epoch, '\t Train loss:', epoch_loss,
                      "\t\t Test loss:",t_loss, '\t\t Test acc:', acc, '\t False pos:', f_p, 'False neg:', 1-acc-f_p)


if __name__ == "__main__":
    print("Loading in data ...")


    bp = "C:\\Users\\raymo\\Desktop\\nn_input\\"
    train_true = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_train_heavy.csv", otherStations=3, previousTimes=2)
    train_false = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_train_non_heavy.csv", otherStations=3, previousTimes=2, nrow=8*len(train_true))
    train_size = len(train_true) + len(train_false)
    print("Train length:", train_size)
    test_true = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_heavy.csv", otherStations=3, previousTimes=2)
    test_false = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_non_heavy.csv", otherStations=3, previousTimes=2, nrow=10*len(test_true))
    test_size = len(test_true) + len(test_false)
    print("Train length:", test_size)

    ## note training is now testing

    print("Initializing NN")
    # learning_rate = 0.001
    n_epoch = 1000 + 1
    display_step = 100
    batch_size = 1024


    x = tf.placeholder(tf.float32, [None, 78])
    y = tf.placeholder(tf.float32)

    train_loss_plot = [[], []]
    test_loss_plot = [[], []]
    test_acc_plot = [[], []]

    print("Starting training")
    train_neural_network(x)

    print("Plotting progress")
    plt.plot(train_loss_plot[0], train_loss_plot[1], label='Train loss')
    plt.plot(test_loss_plot[0], test_loss_plot[1], label='Test loss')
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.title('Cost function')
    plt.show()
    plt.plot(test_acc_plot[0], test_acc_plot[1], label='Test Acc')
    plt.legend(loc='best')
    plt.xlabel("Epoch")
    plt.title("Accuracy")
    plt.show()
