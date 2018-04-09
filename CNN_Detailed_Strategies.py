
# coding: utf-8

# In[21]:


import tensorflow as tf
import numpy as np
import pandas as pd
import random
from scipy.stats import entropy
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import kneighbors_graph

# In[22]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# In[23]:


# make sure the images are np.array type
# checking the shape of the training set
type(mnist.train.images) , mnist.train.images.shape


# In[24]:


# the labels are np.array and already hot encoded
mnist.train.labels


# In[25]:


X_train=mnist.train.images/255.
y_train = mnist.train.labels


# In[26]:


X_test= mnist.test.images/255.
y_test = mnist.test.labels


# In[27]:


X_train.shape ,  y_train.shape ,  X_test.shape  , y_test.shape



regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
is_training = tf.placeholder_with_default(True, shape=[])

# placeholders,hidden layers,
tf.reset_default_graph()

is_training = tf.placeholder_with_default(True, shape=[])
X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')

# input layer
input_layer = tf.reshape(X, [-1, 28, 28, 1])
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
# dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=is_training)


# output layer
y_hat = tf.layers.dense(net, 10, name='y_hat', activation=None, kernel_regularizer=regularizer)


# loss function
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_hat)

lr = tf.placeholder_with_default(0.01, shape=[])
gd = tf.train.AdagradOptimizer(lr)
training_op = tf.contrib.training.create_train_op(loss, gd)


# In[ ]:


# Train the model train 5000 examples
init = tf.global_variables_initializer()

NUM_EX = 55000
BATCH_SIZE = 1000

with tf.Session() as sess:
    init.run()
    for epoch in range(30):
        for i in range(0, NUM_EX, BATCH_SIZE):
            print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
            slice_X = X_train[i:i+BATCH_SIZE]
            slice_y = y_train[i:i+BATCH_SIZE]
            sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
            if i / BATCH_SIZE % 10 == 0:
                training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                predictions, test_loss = sess.run((y_hat, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                print('training loss', training_loss, 'test loss', test_loss, 'test accuracy', test_accuracy)


# In[ ]:
# Random
init = tf.global_variables_initializer()
NUM_EX = 20
BATCH_SIZE = 2
COLLECTIONS = 4
SIZE_SELECTION = 4

first_time = True

SELECTED = list(range(4))
fout=open("random.csv","w")
with tf.Session() as sess:
    init.run()


    for c in range(COLLECTIONS):

        # TAKE WHAT WE HAVE
        x_known = X_train[SELECTED]
        y_known = y_train[SELECTED]


        for epoch in range(30):
            for i in range(0, len(SELECTED), BATCH_SIZE):
                print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
                slice_X = x_known[i:i+BATCH_SIZE]
                slice_y = y_known[i:i+BATCH_SIZE]
                        # import IPython; IPython.embed()
                sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
                if first_time:
                    #import IPython; IPython.embed()
                    first_time = False
                if i / BATCH_SIZE % 10 == 0:
                    training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                    predictions, test_loss = sess.run((y_hat, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                    # import IPython; IPython.embed()
                    test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                    print('training loss', training_loss.mean(), 'test loss', test_loss.mean(), 'test accuracy', test_accuracy)
        fout.write(("%i , %f\n") % (len(SELECTED), test_accuracy))

        # Expand SELECTED
        NOT_SELECTED = [i for i in range(NUM_EX) if i not in SELECTED]
        if len(NOT_SELECTED) > SIZE_SELECTION:
            next_batch= list(random.sample(NOT_SELECTED, SIZE_SELECTION))
        else:
            next_batch=NOT_SELECTED
        SELECTED += next_batch

fout.close()

            # In[55]:
# Least confident
init = tf.global_variables_initializer()
NUM_EX = 20
BATCH_SIZE = 2
COLLECTIONS = 4
SIZE_SELECTION = 4

first_time = True

SELECTED = list(range(4))

fout=open("lc.csv","w")
with tf.Session() as sess:
    init.run()

    for c in range(COLLECTIONS):

        # TAKE WHAT WE HAVE
        x_known = X_train[SELECTED]
        y_known = y_train[SELECTED]


        for epoch in range(30):
            for i in range(0, len(SELECTED), BATCH_SIZE):
                print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
                slice_X = x_known[i:i+BATCH_SIZE]
                slice_y = y_known[i:i+BATCH_SIZE]
                sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
                if first_time:
                    #import IPython; IPython.embed()
                    first_time = False
                if i / BATCH_SIZE % 10 == 0:
                    training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                    predictions, test_loss = sess.run((y_hat, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                    test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                    print('training loss', training_loss.mean(), 'test loss', test_loss.mean(), 'test accuracy', test_accuracy)
        fout.write(("%i , %f\n") % (len(SELECTED), test_accuracy))

        # Expand SELECTED
        NOT_SELECTED = [i for i in range(NUM_EX) if i not in SELECTED]
        predictions = sess.run(y_hat, feed_dict={X: X_train[NOT_SELECTED]})
        max_prob=[ np.max(pred) for pred in predictions]
        store={}
        for i, j in zip(NOT_SELECTED,max_prob):
            store[i] = j
        df=pd.Series(store)
        next_batch= list(df.sort_values(ascending=True)[0: SIZE_SELECTION].index.values)

        # CHANGE
        for k in next_batch:
            SELECTED.append(k)
fout.close()


# Most confident
init = tf.global_variables_initializer()
NUM_EX = 20
BATCH_SIZE = 2
COLLECTIONS = 4
SIZE_SELECTION = 4

first_time = True

SELECTED = list(range(4))

fout=open("mostconf.csv","w")
with tf.Session() as sess:
    init.run()

    for c in range(COLLECTIONS):

        # TAKE WHAT WE HAVE
        x_known = X_train[SELECTED]
        y_known = y_train[SELECTED]


        for epoch in range(30):
            for i in range(0, len(SELECTED), BATCH_SIZE):
                print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
                slice_X = x_known[i:i+BATCH_SIZE]
                slice_y = y_known[i:i+BATCH_SIZE]
                sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
                if first_time:
                    #import IPython; IPython.embed()
                    first_time = False
                if i / BATCH_SIZE % 10 == 0:
                    training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                    predictions, test_loss = sess.run((y_hat, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                    test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                    print('training loss', training_loss.mean(), 'test loss', test_loss.mean(), 'test accuracy', test_accuracy)
        fout.write(("%i , %f\n") % (len(SELECTED), test_accuracy))

        # Expand SELECTED
        NOT_SELECTED = [i for i in range(NUM_EX) if i not in SELECTED]
        predictions = sess.run(y_hat, feed_dict={X: X_train[NOT_SELECTED]})
        max_prob=[ np.max(pred) for pred in predictions]
        store={}
        for i, j in zip(NOT_SELECTED,max_prob):
            store[i] = j
        df=pd.Series(store)
        next_batch= list(df.sort_values(ascending=False)[0: SIZE_SELECTION].index.values)
        x_not_known = X_train[next_batch]
        pred_x_not_known = sess.run(y_hat, feed_dict={X: x_not_known})
        # import IPython; IPython.embed()

        # CHANGE
        for k in next_batch:
            SELECTED.append(k)
fout.close()

# Maximum Entropy
init = tf.global_variables_initializer()
NUM_EX = 20
BATCH_SIZE = 2
COLLECTIONS = 4
SIZE_SELECTION = 4

first_time = True

SELECTED = list(range(4))

fout=open("maxentropy.csv","w")
with tf.Session() as sess:
    init.run()

    for c in range(COLLECTIONS):

        # TAKE WHAT WE HAVE
        x_known = X_train[SELECTED]
        y_known = y_train[SELECTED]


        for epoch in range(30):
            for i in range(0, len(SELECTED), BATCH_SIZE):
                print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
                slice_X = x_known[i:i+BATCH_SIZE]
                slice_y = y_known[i:i+BATCH_SIZE]
                sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
                if first_time:
                    #import IPython; IPython.embed()
                    first_time = False
                if i / BATCH_SIZE % 10 == 0:
                    training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                    predictions, test_loss = sess.run((y_hat, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                    test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                    print('training loss', training_loss.mean(), 'test loss', test_loss.mean(), 'test accuracy', test_accuracy)
        fout.write(("%i , %f\n") % (len(SELECTED), test_accuracy))

        # Expand SELECTED
        NOT_SELECTED = [i for i in range(NUM_EX) if i not in SELECTED]
        y_hat_activated = tf.nn.softmax(y_hat)
        predictions = sess.run(y_hat_activated, feed_dict={X: X_train[NOT_SELECTED]})
        #import IPython; IPython.embed()
        entropy_caluc= [entropy(pred) for pred in predictions]
        store={}
        for i, j in zip(NOT_SELECTED,entropy_caluc):
            store[i] = j
        df=pd.Series(store)
        next_batch= list(df.sort_values(ascending=False)[0: SIZE_SELECTION].index.values)

        # CHANGE
        for k in next_batch:
            SELECTED.append(k)
fout.close()


# graph based
init = tf.global_variables_initializer()
NUM_EX = 20
BATCH_SIZE = 2
COLLECTIONS = 4
SIZE_SELECTION = 4

first_time = True

SELECTED = list(range(4))
fout=open("GraphBasedPairWise.csv","w")
with tf.Session() as sess:
    init.run()


    for c in range(COLLECTIONS):


        # TAKE WHAT WE HAVE
        x_known = X_train[SELECTED]
        y_known = y_train[SELECTED]


        for epoch in range(30):
            for i in range(0, len(SELECTED), BATCH_SIZE):
                print('%i. Training on range %i:%i' % (epoch, i, i+BATCH_SIZE))
                slice_X = x_known[i:i+BATCH_SIZE]
                slice_y = y_known[i:i+BATCH_SIZE]
                sess.run(training_op, feed_dict={X: slice_X, y: slice_y})
                if first_time:
                    first_time = False
                if i / BATCH_SIZE % 10 == 0:
                    training_loss = sess.run(loss, feed_dict={X: slice_X, y: slice_y})
                    predictions, test_loss = sess.run((y_hat_activated, loss), feed_dict={X: X_test[:1000], y: y_test[:1000]})
                    test_accuracy = np.mean(predictions.argmax(axis=1) == y_test[:1000].argmax(axis=1))
                    print('training loss', training_loss.mean(), 'test loss', test_loss.mean(), 'test accuracy', test_accuracy)
        fout.write(("%i , %f\n") % (len(SELECTED), test_accuracy))

        predictions = sess.run(y_hat_activated, feed_dict={X: X_train[:NUM_EX]})

        distance = pairwise_distances(predictions,metric='euclidean')

        for s in range(SIZE_SELECTION):
            df=pd.DataFrame(distance[SELECTED]).min()
            next_one=df.idxmax(axis=0)
            SELECTED.append(next_one)

fout.close()
