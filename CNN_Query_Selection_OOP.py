

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from tensorflow.examples.tutorials.mnist import input_data

# %matplotlib inline


flags.DEFINE_string('output_directory',
                    '/Users/tala/documents/tensorflow/script/output',
                    'CSV files will be saved here.')
flags.DEFINE_string('strategy', "GraphBasedPairWise",
                    'One of: "Random", "MostConfident", "LeastConfident","MaxEntropy","GraphBasedPairWise".')
flags.DEFINE_integer('total_data_size', 2000,
                     'We will only consider subset of MNIST with this size.')
flags.DEFINE_integer('total_test_size', 1000, 'Test dataset size')
flags.DEFINE_integer('num_rating_steps', 400,
                     'Number of times the active learner is allowed to ask for '
                     'ratings.')
flags.DEFINE_integer('num_ratings_per_step', 5,
                     'This many examples will be added at every rating step. '
                     'At the end, total number of rated examples will be '
                     'this flag times --num_rating_steps.')
flags.DEFINE_integer('train_epochs_per_step', 1, 'This many train epochs.')
flags.DEFINE_integer('batch_size', 25, 'Number of examples to be trained on in size of batch size')

# flags.DEFINE_integer('n_neighbors',10,"Number of neighbors for each sample")


FLAGS = flags.FLAGS


def experiment_name():
    return '_'.join([
        's-%s' % FLAGS.strategy,
        'ds-%i' % FLAGS.total_data_size,
        'ts-%i' % FLAGS.total_test_size,
        'p-%i' % FLAGS.num_rating_steps,
        'r-%i' % FLAGS.num_ratings_per_step,
        'e-%i' % FLAGS.train_epochs_per_step,
        'b-%i' % FLAGS.batch_size
        ])


class ActiveLearner(object):
    """Base class for active learning."""

    def __init__(self):
        tf.reset_default_graph()

            # Construct neural network
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        is_training = tf.placeholder_with_default(True, shape=[])

            # placeholders,hidden layers,

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')

            # input layer
        input_layer = tf.reshape(self.x, [-1, 28, 28, 1])
        net=input_layer

            # CNN Layers
        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.conv2d(inputs=net,filters=64, kernel_size=[5, 5], activation=tf.nn.relu, kernel_regularizer=regularizer)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        net = tf.layers.batch_normalization(net, training=is_training)
        net = tf.layers.conv2d(inputs=net, filters=128,kernel_size=[5, 5],activation=tf.nn.relu, kernel_regularizer=regularizer)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

        net=tf.layers.flatten(inputs=net)
        net = tf.layers.batch_normalization(net, training=is_training)
        net=tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)
        self.dense=net
            # output layer
        self.y_hat = tf.layers.dense(net, 10, name='y_hat', activation=None, kernel_regularizer=regularizer)
        self.predictions = tf.nn.softmax(self.y_hat)

            # loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_hat)

        self.lr = tf.placeholder_with_default(0.01, shape=[])
        gd = tf.train.AdagradOptimizer(self.lr)
        self.training_op = tf.contrib.training.create_train_op(loss, gd)

        init=tf.global_variables_initializer()
        self.session= tf.Session()
        self.session.run(init)


    def train_epoch(self,x_train, y_train):
        return self.session.run(self.training_op, feed_dict={self.x: x_train, self.y: y_train})

    def predict(self,x_test):

        return self.session.run(self.y_hat, feed_dict={self.x: x_test})


class RandomActiveLearner(ActiveLearner):

    def __init__(self):
        super().__init__()

    def choose_next_ratings(self, all_x_train, selected_idx, selected_y_train,
                            num_more_ratings):
        all_indices = set(range(len(all_x_train)))
        unselected_indices = all_indices.difference(selected_idx)
        return random.sample(unselected_indices, num_more_ratings)


class MostConfidetnActiveLearner(ActiveLearner):

    def __init__(self):
        super().__init__()

    def choose_next_ratings(self, all_x_train, selected_idx, selected_y_train,
                            num_more_ratings):
        all_indices = set(range(len(all_x_train)))
        unselected_indices = list(all_indices.difference(selected_idx))

        predictions = self.session.run(
            self.predictions, feed_dict={self.x: all_x_train[unselected_indices]})


        select = []
        for j, p in sorted(enumerate(predictions.max(axis=1)), key=lambda x: -x[1])[:num_more_ratings]:
            select.append(unselected_indices[j])
        return select

class LeastConfidentActiveLearner(ActiveLearner):

    def __init__(self):
        super().__init__()

    def choose_next_ratings(self, all_x_train, selected_idx, selected_y_train,num_more_ratings):

        all_indices = set(range(len(all_x_train)))
        unselected_indices = list(all_indices.difference(selected_idx))

        predictions = self.session.run(
            self.predictions, feed_dict={self.x: all_x_train[unselected_indices]})


        select = []
        for j, p in sorted(enumerate(predictions.max(axis=1)), key=lambda x: x[1])[:num_more_ratings]:
            select.append(unselected_indices[j])
        return select

class MaximumEntropyActiveLearner(ActiveLearner):

    def __init__(self):
        super().__init__()

    def choose_next_ratings(self, all_x_train, selected_idx, selected_y_train,
                            num_more_ratings):
        all_indices = set(range(len(all_x_train)))
        unselected_indices = list(all_indices.difference(selected_idx))

        predictions = self.session.run(self.predictions, feed_dict={self.x: all_x_train[unselected_indices]})
        entropy_caluc= [entropy(pred) for pred in predictions]
        select = []
        for j, p in sorted(enumerate(entropy_caluc), key=lambda x: -x[1])[:num_more_ratings]:
            select.append(unselected_indices[j])
        return select

class GraphBasedPairWiseActiveLearner(ActiveLearner):

    def __init__(self):
        super().__init__()
        self.exported_once = False

    def choose_next_ratings(self, all_x_train, selected_idx, selected_y_train,num_more_ratings):

        predictions = self.session.run(self.predictions, feed_dict={self.x:  all_x_train})
        distance = pairwise_distances(predictions,metric='euclidean')

        if not self.exported_once:
            exp_name = experiment_name()
            np_file_path = os.path.join(FLAGS.output_directory, exp_name + 'distances')
            np.save(np_file_path, distance)
            self.exported_once = True

        select = []
        for s in range(num_more_ratings):
            df= pd.DataFrame(distance[selected_idx + select]).min()
            next_one=df.idxmax(axis=0)
            select.append(next_one)
        #import IPython; IPython.embed()
        return select


LEARNER_CLASSES = {
    'Random': RandomActiveLearner,
    'MostConfident': MostConfidetnActiveLearner,
    'LeastConfident': LeastConfidentActiveLearner,
    'MaxEntropy': MaximumEntropyActiveLearner,
    'GraphBasedPairWise': GraphBasedPairWiseActiveLearner,

}


def main(argv):
    learner_class = LEARNER_CLASSES[FLAGS.strategy]

    # Get the dataset.
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x_train=mnist.train.images/255.
    y_train = mnist.train.labels
    x_train = x_train[:FLAGS.total_data_size]
    y_train = y_train[:FLAGS.total_data_size]

    x_test = mnist.test.images/255.
    y_test = mnist.test.labels

    x_test = x_test[:FLAGS.total_test_size]
    y_test = y_test[:FLAGS.total_test_size]

    #

    learner = learner_class()

    selected_idx = list(range(FLAGS.num_ratings_per_step))

    total_epochs = FLAGS.num_rating_steps * FLAGS.train_epochs_per_step
    done_epochs = 0
    exp_name = experiment_name()
    csv_file_path = os.path.join(FLAGS.output_directory, exp_name + '.csv')

    csv_out = open(csv_file_path, 'w')
    csv_out.write("Number of examples,Accuracy Score\n")




    for rating_step in range(FLAGS.num_rating_steps):
        for epoch in range(FLAGS.train_epochs_per_step):
            learner.train_epoch(x_train[selected_idx], y_train[selected_idx])
            print

        done_epochs += 1
        predictions = learner.predict(x_test)
        accuracy = np.mean(predictions.argmax(axis=1) == y_test.argmax(axis=1))
        print('finished %i%%. Accuracy=%f' % (100 * done_epochs / total_epochs, accuracy))
        csv_out.write('%i,%f\n' % (len(selected_idx), accuracy))

        next_indices = learner.choose_next_ratings(
                x_train, selected_idx, y_train[selected_idx],
                FLAGS.num_ratings_per_step)
        
        assert len(next_indices) == FLAGS.num_ratings_per_step
        selected_idx += next_indices
    csv_out.close()

if __name__ == '__main__':
    app.run(main)
