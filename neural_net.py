import tensorflow as tf
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from datetime import datetime


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

    # distance (0-20) (rel, location)
    set = np.arange(0, otherStations * 2 + 0, 2)
    data[:, set] /= 20



    # directions (0-360) (rel location, main wind dir)
    set = np.concatenate((np.arange(1,otherStations*2 + 1, 2), np.arange(otherStations * 2 + 4, otherStations*2 + 4 +  (1 + otherStations)*6*(previousTimes + 1), 6)))
    scaleVal = (285.87 - 102.37)
    data[:, set] -= 102.37
    data[:, set] /= scaleVal


    # air temperature main station
    set = np.arange(otherStations * 2 + 0, otherStations*2 + 0 + (1 + otherStations)*6*(previousTimes + 1), 6)
    scaleVal = (18.04 - 5.23)
    data[:, set] -= 5.23
    data[:, set] /= scaleVal


    # relative humidity 0->1
    set = np.arange(otherStations * 2 + 1, otherStations * 2 + 1 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    scaleVal = (95.35 - 66.36)
    data[:, set] -= 66.36
    data[:, set] /= scaleVal

    # vapor pressure 0->1
    set = np.arange(otherStations * 2 + 2, otherStations * 2 + 2 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    scaleVal = (1.55 - 0.74) / 2
    data[:, set] -= 0.74
    data[:, set] /= scaleVal

    # wind speed 0->1
    set = np.arange(otherStations * 2 + 3, otherStations * 2 + 3 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    scaleVal = (3.15 - 0.62)
    data[:, set] -= 0.62
    data[:, set] /= scaleVal

    # rain 0 -> 1
    set = np.arange(otherStations * 2 + 5, otherStations * 2 + 5 + (1 + otherStations) * 6 * (previousTimes + 1), 6)
    data[:, set] /= 0.57

    set = []
    # Tair 0, RH 1, vapor_pressure_{Avg} 2, WindSpd_{Avg} 3, WindDir_{Avg} 4, Rain_{Tot} 5
    '''
    # remove a specific output
    ri = [0, 1]
    for i in ri:
        set = np.concatenate((set,
                              np.arange(otherStations * 2 + i, otherStations * 2 + i + (1 + otherStations) * 6 * (previousTimes + 1), 6)
                              ))
    '''

    '''
    # remove other stations
    set = np.concatenate((set, np.arange(0, 6))) # relative locations
    for i in range(previousTimes+1):
        set = np.concatenate((
            set, np.arange(6 + i*(1+otherStations)*6, 6 + i*(1+otherStations)*6 + otherStations*6)
        ))
    #'''

    # time stamps
    # set = np.arange(30, 78)

    data = np.delete(data, np.s_[set], 1)

    return data


def get_train_batch(batch_size):
    n_true = int(batch_size / (train_size/len(train_true)))
    n_false = batch_size - n_true

    n_true_sel = np.random.randint(0, len(train_true), size=n_true)
    n_false_sel = np.random.randint(0, len(train_false), size=n_false)


    x = np.concatenate((
        train_true[n_true_sel], train_false[n_false_sel]
    ))
    y = np.zeros(batch_size)

    y[:n_true] = 1.0
    #y[n_true:, 1] = 1


    return x, y


def get_test_batch(batch_size=None): # send all
    if (batch_size == None):
        batch_size = test_size
    n_true = int(batch_size / (test_size/len(test_true)))
    n_false = batch_size - n_true

    n_true_sel = np.random.randint(0, len(test_true), size=n_true)
    n_false_sel = np.random.randint(0, len(test_false), size=n_false)

    x = np.concatenate((
        test_true[n_true_sel], test_false[n_false_sel]
    ))
    y = np.zeros(batch_size)
    y[:n_true] = 1.0
    #y[n_true:, 1] = 1

    return x, y


def neural_network_model(input_feed, input_size, layer_size):
    n_layer = 0
    var_count = 0
    tfVar = []
    layer = [input_feed]
    prevLay = input_size
    for size in layer_size:
        tfVar.append({
            'weights': tf.Variable(tf.random_normal([prevLay, size])),
            'biases': tf.Variable(tf.random_normal([size]))
        })

        layer.append(tf.nn.sigmoid(layer[n_layer] @ tfVar[n_layer]['weights'] + tfVar[n_layer]['biases']))
        var_count += (1 + prevLay) * size
        prevLay = size
        n_layer += 1

    print(n_layer-1, 'hidden layers, containing', var_count, 'variables')
    return layer[-1], tfVar


def train_neural_network(prediction_sigmoid, pos_weight=1):
    weighted_loss = tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=prediction_sigmoid, pos_weight=pos_weight)
    cost = tf.reduce_mean(weighted_loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    predicted_class = tf.greater_equal(prediction_sigmoid, 0.5)
    correct = tf.equal(predicted_class, tf.equal(y, 1.0))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epoch):
            epoch_loss = 0
            for _ in range(int(train_size/batch_size)):
                epoch_x, epoch_y = get_train_batch(batch_size)
                _, c, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            epoch_loss /= int(train_size/batch_size)

            train_loss_plot[0].append(epoch)
            train_loss_plot[1].append(epoch_loss)

            if epoch % display_step == 0:
                test_x, test_y = get_test_batch(1024) # way to small but first trying to reduce training loss more
                t_loss, acc = sess.run([cost, accuracy], feed_dict={x: test_x, y: test_y})

                heavy_rain_acc = accuracy.eval({x: test_true, y: test_true_y})  # TP/ FN

                true_pos = np.round(heavy_rain_acc*len(test_true))
                false_neg = len(test_true) - true_pos
                true_neg = np.round((len(test_true) + len(test_false))*acc) - true_pos
                false_pos = len(test_false) - true_neg

                test_loss_plot[0].append(epoch)
                test_loss_plot[1].append(t_loss)
                test_acc_plot[0].append(epoch)
                test_acc_plot[1].append(acc)
                print('Epoch', epoch, '\t Train loss:', epoch_loss, '\t acc: ', train_acc,
                      "\t\t Test loss:",t_loss, '\t\tacc:', acc,
                      '\t heavy rain acc:', heavy_rain_acc,
                      '\tTP:', true_pos, 'FN:', false_neg, 'TN:', true_neg, 'FP:', false_pos
                      )


if __name__ == "__main__":
    print("Loading in data ...")

    nNoHeavy = 2
    print('Ratio not-heavy:heavy', nNoHeavy)

    # Constant variables based on data_management.py
    bp = "C:\\Users\\raymo\\Desktop\\nn_input\\"
    train_true = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_train_heavy.csv",
                                         otherStations=3, previousTimes=2)
    train_false = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_train_non_heavy.csv",
                                          otherStations=3, previousTimes=2, nrow=nNoHeavy*len(train_true))
    train_size = len(train_true) + len(train_false)
    print("Train length:", train_size, 'ratio:', len(train_true)/train_size)
    test_true = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_heavy.csv",
                                        otherStations=3, previousTimes=2)
    test_false = load_and_normalize_feed(bp + "raw_3+1_station_-10+15min_test_non_heavy.csv",
                                         otherStations=3, previousTimes=2)
    test_size = len(test_true) + len(test_false)
    print("Test length:", test_size, 'ratio:', len(test_true) / test_size)
    test_true_y = np.ones(len(test_true))

    # hyper parameters
    n_epoch = 500 + 1
    display_step = 50
    batch_size = 128

    n_input = len(test_true[0])
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32)

    print(n_input, 'inputs')

    arr_node_conf = [[40, 15, 1]]
    ''',
                     [50, 5, 2],
                     [100, 2],
                     [30, 2]
                     ]
    '''
    for i in range(len(arr_node_conf)):
        print("Initializing NN", arr_node_conf[i])

        train_loss_plot = [[], []]
        test_loss_plot = [[], []]
        test_acc_plot = [[], []]

        print("Starting training, at", datetime.now())
        prediction, variables = neural_network_model(x, n_input, arr_node_conf[i])
        train_neural_network(prediction, 1)

        print("Plotting progress")
        plt.subplot(len(arr_node_conf), 2, i*2+1)
        plt.plot(train_loss_plot[0], train_loss_plot[1], label='Train loss ' + str(arr_node_conf[i]))
        plt.plot(test_loss_plot[0], test_loss_plot[1], label='Test loss ' + str(arr_node_conf[i]))
        plt.xlabel("Epoch")
        plt.legend(loc='best')
        plt.subplot(len(arr_node_conf), 2, i*2 + 2)
        plt.plot(test_acc_plot[0], test_acc_plot[1], label='Test Acc ' + str(arr_node_conf[i]))
        plt.legend(loc='best')
        plt.xlabel("Epoch")

    plt.show()
    print("Done")