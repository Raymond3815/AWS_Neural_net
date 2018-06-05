'''
import station_data as sd
from scipy.stats import pearsonr

data = sd.loadStationToDict(['Oost'], False)['Oost']['data'].dropna()

for c in data.keys():
    r, p = pearsonr(data[c], data['vapor_pressure_{Avg}'])
    print(c, 'to vp intensity R =', r, ', R*R =', r**2, '(p =', p, ')')
'''
'''
import numpy as np

arr = [[1,0,-1], [1,1,1], [1,2,4]]
print(np.sum(np.abs(arr), axis=1))
'''
'''
import matplotlib.pyplot as plt


def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
   
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                ax.add_artist(line)

#fig = plt.figure(figsize=(12, 12))
#draw_neural_net(fig.gca(), .1, .9, .1, .9, [2, 6, 4, 3])
#plt.show()
'''


import numpy as np
import tensorflow as tf

y = tf.placeholder(tf.float32)
r = tf.cast(tf.Variable(tf.random_uniform([10000])), tf.float32)

nY = np.ones(10000)

pred_class = tf.greater_equal(r, 0.5)
corr = tf.equal(pred_class, tf.greater_equal(y, 0.5))
acc = tf.reduce_mean(tf.cast(corr, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(acc.eval({y: nY}))

