import tensorflow as tf
import time
import random
from tensorflow.examples.tutorials.mnist import input_data
#tf.

ctime = time.time()
ttime = 0
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 1e-3
training_epochs = 15
batch_size = 10000
std_dev = 0.1

X = tf.placeholder(tf.float32, [None, 28*28])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

w_i = tf.Variable(tf.random_normal([28*28, 20*20], stddev = std_dev))
b_i = tf.Variable(tf.random_normal([20*20]))

w_1 = tf.Variable(tf.random_normal([20*20, 15*15], stddev = std_dev))
b_1 = tf.Variable(tf.random_normal([15*15]))

w_2 = tf.Variable(tf.random_normal([15*15, 10*10], stddev = std_dev))
b_2 = tf.Variable(tf.random_normal([10*10]))

w_3 = tf.Variable(tf.random_normal([10*10, 8*8], stddev = std_dev))
b_3 = tf.Variable(tf.random_normal([8*8]))

w_o = tf.Variable(tf.random_normal([8*8, 10], stddev = std_dev))
b_o = tf.Variable(tf.random_normal([10]))


L_i = tf.nn.relu(tf.matmul(X, w_i) + b_i)
L_1 = tf.nn.relu(tf.matmul(L_i, w_1) + b_1)
L_2 = tf.nn.relu(tf.matmul(L_1, w_2) + b_2)
L_3 = tf.nn.relu(tf.matmul(L_2, w_3) + b_3)
L_o = tf.matmul(L_3, w_o) + b_o

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = L_o))
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

    correct_prediction = tf.equal(tf.argmax(L_o, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('Accuracy : ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label : ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print('Prediction : ', sess.run(tf.argmax(L_o, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

'''
batch size = 15 , epoch = 15
epoch : 1/15
process time : 36.739774 sec / total time : 36.739774 sec
Epoch :  0001 cost =  0.227206940
epoch : 2/15
process time : 46.854998 sec / total time : 83.594772 sec
Epoch :  0002 cost =  0.106411636
epoch : 3/15
process time : 81.782035 sec / total time : 165.376807 sec
Epoch :  0003 cost =  0.077120085
epoch : 4/15
process time : 94.921098 sec / total time : 260.297906 sec
Epoch :  0004 cost =  0.062847946
epoch : 5/15
process time : 91.051860 sec / total time : 351.349766 sec
Epoch :  0005 cost =  0.051689128
epoch : 6/15
process time : 78.547123 sec / total time : 429.896888 sec
Epoch :  0006 cost =  0.041963816
epoch : 7/15
process time : 67.247929 sec / total time : 497.144817 sec
Epoch :  0007 cost =  0.035786827
epoch : 8/15
process time : 46.994845 sec / total time : 544.139663 sec
Epoch :  0008 cost =  0.032138194
epoch : 9/15
process time : 45.454483 sec / total time : 589.594145 sec
Epoch :  0009 cost =  0.031354821
epoch : 10/15
process time : 45.589018 sec / total time : 635.183163 sec
Epoch :  0010 cost =  0.027276233
epoch : 11/15
process time : 46.442659 sec / total time : 681.625822 sec
Epoch :  0011 cost =  0.026874786
epoch : 12/15
process time : 48.532790 sec / total time : 730.158612 sec
Epoch :  0012 cost =  0.024082519
epoch : 13/15
process time : 48.679176 sec / total time : 778.837788 sec
Epoch :  0013 cost =  0.022623313
epoch : 14/15
process time : 47.293594 sec / total time : 826.131382 sec
Epoch :  0014 cost =  0.019317338
epoch : 15/15
process time : 46.404347 sec / total time : 872.535729 sec
Epoch :  0015 cost =  0.019718877
Learning Finished!
Accuracy :  0.98
Label :  [4]
Prediction :  [4]

Process finished with exit code 0
'''

'''
batch_size : 100
epoch : 1/15
process time : 9.956136 sec / total time : 9.956136 sec
Epoch :  0001 cost =  0.302832699
epoch : 2/15
process time : 8.194076 sec / total time : 18.150212 sec
Epoch :  0002 cost =  0.107401087
epoch : 3/15
process time : 8.476033 sec / total time : 26.626245 sec
Epoch :  0003 cost =  0.068210179
epoch : 4/15
process time : 10.050001 sec / total time : 36.676246 sec
Epoch :  0004 cost =  0.048970376
epoch : 5/15
process time : 11.141374 sec / total time : 47.817620 sec
Epoch :  0005 cost =  0.035208502
epoch : 6/15
process time : 15.381491 sec / total time : 63.199111 sec
Epoch :  0006 cost =  0.031064544
epoch : 7/15
process time : 13.757993 sec / total time : 76.957103 sec
Epoch :  0007 cost =  0.025209733
epoch : 8/15
process time : 11.678393 sec / total time : 88.635497 sec
Epoch :  0008 cost =  0.026078354
epoch : 9/15
process time : 15.773012 sec / total time : 104.408509 sec
Epoch :  0009 cost =  0.015775953
epoch : 10/15
process time : 16.395014 sec / total time : 120.803524 sec
Epoch :  0010 cost =  0.021233605
epoch : 11/15
process time : 17.275004 sec / total time : 138.078527 sec
Epoch :  0011 cost =  0.015883474
epoch : 12/15
process time : 17.875143 sec / total time : 155.953670 sec
Epoch :  0012 cost =  0.014788332
epoch : 13/15
process time : 18.464677 sec / total time : 174.418347 sec
Epoch :  0013 cost =  0.014176505
epoch : 14/15
process time : 20.511687 sec / total time : 194.930034 sec
Epoch :  0014 cost =  0.011893390
epoch : 15/15
process time : 19.408596 sec / total time : 214.338631 sec
Epoch :  0015 cost =  0.013476321
Learning Finished!
Accuracy :  0.9801
Label :  [8]
Prediction :  [8]
'''

'''
batch_size : 1000
epoch : 1/15
process time : 6.509087 sec / total time : 6.509087 sec
Epoch :  0001 cost =  0.934255416
epoch : 2/15
process time : 5.457389 sec / total time : 11.966475 sec
Epoch :  0002 cost =  0.222120689
epoch : 3/15
process time : 5.255153 sec / total time : 17.221628 sec
Epoch :  0003 cost =  0.147464953
epoch : 4/15
process time : 5.299323 sec / total time : 22.520951 sec
Epoch :  0004 cost =  0.106742145
epoch : 5/15
process time : 6.349986 sec / total time : 28.870937 sec
Epoch :  0005 cost =  0.081228622
epoch : 6/15
process time : 6.915457 sec / total time : 35.786395 sec
Epoch :  0006 cost =  0.065007115
epoch : 7/15
process time : 7.882307 sec / total time : 43.668701 sec
Epoch :  0007 cost =  0.046658999
epoch : 8/15
process time : 7.919698 sec / total time : 51.588399 sec
Epoch :  0008 cost =  0.038761757
epoch : 9/15
process time : 7.546631 sec / total time : 59.135030 sec
Epoch :  0009 cost =  0.029745450
epoch : 10/15
process time : 7.520764 sec / total time : 66.655794 sec
Epoch :  0010 cost =  0.023442272
epoch : 11/15
process time : 7.207324 sec / total time : 73.863118 sec
Epoch :  0011 cost =  0.020264562
epoch : 12/15
process time : 6.996482 sec / total time : 80.859600 sec
Epoch :  0012 cost =  0.012932700
epoch : 13/15
process time : 6.949446 sec / total time : 87.809046 sec
Epoch :  0013 cost =  0.010153593
epoch : 14/15
process time : 6.969574 sec / total time : 94.778619 sec
Epoch :  0014 cost =  0.007144814
epoch : 15/15
process time : 6.924371 sec / total time : 101.702990 sec
Epoch :  0015 cost =  0.004997987
Learning Finished!
Accuracy :  0.9789
Label :  [8]
Prediction :  [8]
'''

'''
batch_size : 10000
epoch : 1/15
process time : 7.579497 sec / total time : 7.579497 sec
Epoch :  0001 cost =  3.089363289
epoch : 2/15
process time : 6.582680 sec / total time : 14.162178 sec
Epoch :  0002 cost =  2.051601934
epoch : 3/15
process time : 6.676253 sec / total time : 20.838431 sec
Epoch :  0003 cost =  1.570296574
epoch : 4/15
process time : 6.276482 sec / total time : 27.114913 sec
Epoch :  0004 cost =  1.088669634
epoch : 5/15
process time : 6.094850 sec / total time : 33.209763 sec
Epoch :  0005 cost =  0.726472616
epoch : 6/15
process time : 6.177175 sec / total time : 39.386938 sec
Epoch :  0006 cost =  0.526521921
epoch : 7/15
process time : 6.368541 sec / total time : 45.755479 sec
Epoch :  0007 cost =  0.419503188
epoch : 8/15
process time : 6.128858 sec / total time : 51.884337 sec
Epoch :  0008 cost =  0.361520779
epoch : 9/15
process time : 7.642481 sec / total time : 59.526818 sec
Epoch :  0009 cost =  0.316171634
epoch : 10/15
process time : 7.987691 sec / total time : 67.514509 sec
Epoch :  0010 cost =  0.281876314
epoch : 11/15
process time : 6.019294 sec / total time : 73.533803 sec
Epoch :  0011 cost =  0.258068055
epoch : 12/15
process time : 6.216920 sec / total time : 79.750722 sec
Epoch :  0012 cost =  0.232850188
epoch : 13/15
process time : 5.986770 sec / total time : 85.737492 sec
Epoch :  0013 cost =  0.212795988
epoch : 14/15
process time : 6.093346 sec / total time : 91.830838 sec
Epoch :  0014 cost =  0.196973529
epoch : 15/15
process time : 6.136863 sec / total time : 97.967701 sec
Epoch :  0015 cost =  0.184537438
Learning Finished!
Accuracy :  0.9473
Label :  [9]
Prediction :  [9]
'''