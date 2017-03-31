import tensorflow as tf
import random
import time

from tensorflow.examples.tutorials.mnist import input_data

ctime = time.time()
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 1e-3
training_epochs = 15
batch_size = 100
std_dev = 0.1

#drop_out = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([28*28, 14*14], stddev = std_dev))
B1 = tf.Variable(tf.random_normal([14*14]))

W2 = tf.Variable(tf.random_normal([14*14, 7*7], stddev = std_dev))
B2 = tf.Variable(tf.random_normal([7*7]))

W3 = tf.Variable(tf.random_normal([7*7, 4*4], stddev = std_dev))
B3 = tf.Variable(tf.random_normal([4*4]))

W4 = tf.Variable(tf.random_normal([4*4, 10], stddev = std_dev))
B4 = tf.Variable(tf.random_normal([10]))

inputLayer = tf.matmul(X, W1) + B1
inputLayer = tf.nn.relu(inputLayer)

L1 = tf.matmul(inputLayer, W2) + B2
L1 = tf.nn.relu(L1)

L2 = tf.matmul(L1, W3) + B3
L2 = tf.nn.relu(L2)

logits = tf.matmul(L2, W4) + B4

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ttime = 0.0
    for epoch in range(training_epochs):
        print('epoch : %d/%d' % (epoch + 1, training_epochs))
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
            avg_cost += c / total_batch

        ttime += time.time() - ctime
        print('process time : %f sec / total time : %f sec' % ((time.time() - ctime) ,ttime))
        ctime = time.time()
        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Accuracy : ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label : ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print('Prediction : ', sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

'''
epoch : 1/15
process time : 4.499079 sec / total time : 4.499079 sec
Epoch :  0001 cost =  0.547301435
epoch : 2/15
process time : 3.428511 sec / total time : 7.927590 sec
Epoch :  0002 cost =  0.168373056
epoch : 3/15
process time : 3.469382 sec / total time : 11.396971 sec
Epoch :  0003 cost =  0.115006735
epoch : 4/15
process time : 3.569551 sec / total time : 14.966523 sec
Epoch :  0004 cost =  0.084689914
epoch : 5/15
process time : 4.385185 sec / total time : 19.351707 sec
Epoch :  0005 cost =  0.066123262
epoch : 6/15
process time : 4.070717 sec / total time : 23.422424 sec
Epoch :  0006 cost =  0.052627339
epoch : 7/15
process time : 4.561757 sec / total time : 27.984181 sec
Epoch :  0007 cost =  0.040992957
epoch : 8/15
process time : 4.515356 sec / total time : 32.499537 sec
Epoch :  0008 cost =  0.032761111
epoch : 9/15
process time : 4.345759 sec / total time : 36.845296 sec
Epoch :  0009 cost =  0.026653379
epoch : 10/15
process time : 3.993336 sec / total time : 40.838632 sec
Epoch :  0010 cost =  0.022022481
epoch : 11/15
process time : 4.056640 sec / total time : 44.895272 sec
Epoch :  0011 cost =  0.018331727
epoch : 12/15
process time : 4.065166 sec / total time : 48.960438 sec
Epoch :  0012 cost =  0.014153740
epoch : 13/15
process time : 3.981394 sec / total time : 52.941833 sec
Epoch :  0013 cost =  0.015116131
epoch : 14/15
process time : 3.894822 sec / total time : 56.836655 sec
Epoch :  0014 cost =  0.012070825
epoch : 15/15
process time : 4.021408 sec / total time : 60.858063 sec
Epoch :  0015 cost =  0.008580195
Learning Finished!
Accuracy :  0.9736
Label :  [7]
Prediction :  [7]
'''


