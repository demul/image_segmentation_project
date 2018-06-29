import data_load
import util
import numpy as np
import tensorflow as tf

class FCN:
    def __init__(self, input_size, lr):
        self.lr = lr
        self.input_size = input_size # conv2d_transpose에서 아웃풋의 형태를 Dynamic하게(None이나 -1로) 지정해 줄 수가 없다.
                                     # 그래서 그냥 아예 이렇게 받아놓는게 나은 것 같다.


    def build(self, input, label, keep_prob=0.5, is_training=False):
        ############################다운샘플#######################
        W1 = tf.Variable(tf.random_normal([3, 3, input.shape[3].value, 64], stddev=0.01), dtype=np.float32)
        L1 = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
        b1 = tf.Variable(tf.random_normal([64]), dtype=np.float32)
        L1 = tf.nn.bias_add(L1, b1)
        L1 = tf.nn.relu(L1)

        W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01), dtype=np.float32)
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        b2 = tf.Variable(tf.random_normal([64]), dtype=np.float32)
        L2 = tf.nn.bias_add(L2, b2)
        L2 = tf.layers.batch_normalization(L2, training=is_training)
        L2 = tf.nn.relu(L2)

        pool1 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), dtype=np.float32)
        L3 = tf.nn.conv2d(pool1, W3, strides=[1, 1, 1, 1], padding='SAME')
        b3 = tf.Variable(tf.random_normal([128]), dtype=np.float32)
        L3 = tf.nn.bias_add(L3, b3)
        L3 = tf.layers.batch_normalization(L3, training=is_training)
        L3 = tf.nn.relu(L3)

        W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01), dtype=np.float32)
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        b4 = tf.Variable(tf.random_normal([128]), dtype=np.float32)
        L4 = tf.nn.bias_add(L4, b4)
        L4 = tf.layers.batch_normalization(L4, training=is_training)
        L4 = tf.nn.relu(L4)

        pool2 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W5 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01), dtype=np.float32)
        L5 = tf.nn.conv2d(pool2, W5, strides=[1, 1, 1, 1], padding='SAME')
        b5 = tf.Variable(tf.random_normal([256]), dtype=np.float32)
        L5 = tf.nn.bias_add(L5, b5)
        L5 = tf.layers.batch_normalization(L5, training=is_training)
        L5 = tf.nn.relu(L5)

        W6 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01), dtype=np.float32)
        L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
        b6 = tf.Variable(tf.random_normal([256]), dtype=np.float32)
        L6 = tf.nn.bias_add(L6, b6)
        L6 = tf.layers.batch_normalization(L6, training=is_training)
        L6 = tf.nn.relu(L6)

        W7 = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=0.01), dtype=np.float32)
        L7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
        b7 = tf.Variable(tf.random_normal([256]), dtype=np.float32)
        L7 = tf.nn.bias_add(L7, b7)
        L7 = tf.layers.batch_normalization(L7, training=is_training)
        L7 = tf.nn.relu(L7)

        pool3 = tf.nn.max_pool(L7, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W8 = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01), dtype=np.float32)
        L8 = tf.nn.conv2d(pool3, W8, strides=[1, 1, 1, 1], padding='SAME')
        b8 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L8 = tf.nn.bias_add(L8, b8)
        L8 = tf.layers.batch_normalization(L8, training=is_training)
        L8 = tf.nn.relu(L8)

        W9 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01), dtype=np.float32)
        L9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME')
        b9 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L9 = tf.nn.bias_add(L9, b9)
        L9 = tf.layers.batch_normalization(L9, training=is_training)
        L9 = tf.nn.relu(L9)

        W10 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01), dtype=np.float32)
        L10 = tf.nn.conv2d(L9, W10, strides=[1, 1, 1, 1], padding='SAME')
        b10 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L10 = tf.nn.bias_add(L10, b10)
        L10 = tf.layers.batch_normalization(L10, training=is_training)
        L10 = tf.nn.relu(L10)

        pool4 = tf.nn.max_pool(L10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W11 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01), dtype=np.float32)
        L11 = tf.nn.conv2d(pool4, W11, strides=[1, 1, 1, 1], padding='SAME')
        b11 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L11 = tf.nn.bias_add(L11, b11)
        L11 = tf.layers.batch_normalization(L11, training=is_training)
        L11 = tf.nn.relu(L11)

        W12 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01), dtype=np.float32)
        L12 = tf.nn.conv2d(L11, W12, strides=[1, 1, 1, 1], padding='SAME')
        b12 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L12 = tf.nn.bias_add(L12, b12)
        L12 = tf.layers.batch_normalization(L12, training=is_training)
        L12 = tf.nn.relu(L12)

        W13 = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=0.01), dtype=np.float32)
        L13 = tf.nn.conv2d(L12, W13, strides=[1, 1, 1, 1], padding='SAME')
        b13 = tf.Variable(tf.random_normal([512]), dtype=np.float32)
        L13 = tf.nn.bias_add(L13, b13)
        L13 = tf.layers.batch_normalization(L13, training=is_training)
        L13 = tf.nn.relu(L13)

        pool5 = tf.nn.max_pool(L13, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        ###########################################################

        #############################1x1 conv########################
        W14 = tf.Variable(tf.random_normal([1, 1, 512, 4096], stddev=0.01), dtype=np.float32)
        FCL1 = tf.nn.conv2d(pool5, W14, strides=[1, 1, 1, 1], padding='SAME')
        b14 = tf.Variable(tf.random_normal([4096]), dtype=np.float32)
        FCL1 = tf.nn.bias_add(FCL1, b14)
        FCL1 = tf.layers.batch_normalization(FCL1, training=is_training)
        FCL1 = tf.nn.relu(FCL1)


        FCL1 = tf.nn.dropout(FCL1, keep_prob=keep_prob)

        W15 = tf.Variable(tf.random_normal([1, 1, 4096, 4096], stddev=0.01), dtype=np.float32)
        FCL2 = tf.nn.conv2d(FCL1, W15, strides=[1, 1, 1, 1], padding='SAME')
        b15 = tf.Variable(tf.random_normal([4096]), dtype=np.float32)
        FCL2 = tf.nn.bias_add(FCL2, b15)
        FCL2 = tf.layers.batch_normalization(FCL2, training=is_training)
        FCL2 = tf.nn.relu(FCL2)


        FCL2 = tf.nn.dropout(FCL2, keep_prob=keep_prob)

        W16 = tf.Variable(tf.random_normal([1, 1, 4096, label.shape[3].value], stddev=0.01), dtype=np.float32)
        FCL3 = tf.nn.conv2d(FCL2, W16, strides=[1, 1, 1, 1], padding='SAME')
        b16 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        FCL3 = tf.nn.bias_add(FCL3, b16)
        FCL3 = tf.layers.batch_normalization(FCL3, training=is_training)
        FCL3 = tf.nn.relu(FCL3)


        FCL3 = tf.nn.dropout(FCL3, keep_prob=keep_prob)
        ###########################################################

        #############################업샘플########################

        ###########skip 연결 from pool4
        SW1 = tf.Variable(tf.random_normal([1, 1, 512, label.shape[3].value], stddev=0.01), dtype=np.float32)
        FCL4_from_pool4 = tf.nn.conv2d(pool4, SW1, strides=[1, 1, 1, 1], padding='SAME')
        Sb1 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        FCL4_from_pool4 = tf.nn.bias_add(FCL4_from_pool4, Sb1)
        FCL4_from_pool4 = tf.layers.batch_normalization(FCL4_from_pool4, training=is_training)
        FCL4_from_pool4 = tf.nn.relu(FCL4_from_pool4)

        ###########업샘플 1
        W17 = tf.Variable(tf.random_normal([4, 4, label.shape[3].value, label.shape[3].value], stddev=0.01),
                          dtype=np.float32)
        UCL1 = tf.nn.conv2d_transpose(FCL3, W17, output_shape=[self.input_size, FCL4_from_pool4.shape[1].value, FCL4_from_pool4.shape[2].value,
                                                               label.shape[3].value], strides=[1, 2, 2, 1],
                                      padding='SAME')
        b17 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        UCL1 = tf.nn.bias_add(UCL1, b17)
        UCL1 = tf.layers.batch_normalization(UCL1, training=is_training)
        UCL1 = tf.nn.leaky_relu(UCL1)



        UCL2 = tf.add(UCL1, FCL4_from_pool4)

        ###########skip 연결 from pool3
        SW2 = tf.Variable(tf.random_normal([1, 1, 256, label.shape[3].value], stddev=0.01), dtype=np.float32)
        FCL5_from_pool3 = tf.nn.conv2d(pool3, SW2, strides=[1, 1, 1, 1], padding='SAME')
        Sb2 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        FCL5_from_pool3 = tf.nn.bias_add(FCL5_from_pool3, Sb2)
        FCL5_from_pool3 = tf.layers.batch_normalization(FCL5_from_pool3, training=is_training)
        FCL5_from_pool3 = tf.nn.relu(FCL5_from_pool3)
        ###########업샘플 2
        W18 = tf.Variable(tf.random_normal([4, 4, label.shape[3].value, label.shape[3].value], stddev=0.01),
                          dtype=np.float32)
        UCL3 = tf.nn.conv2d_transpose(UCL2, W18, output_shape=[self.input_size, FCL5_from_pool3.shape[1].value, FCL5_from_pool3.shape[2].value,
                                                               label.shape[3].value], strides=[1, 2, 2, 1],
                                      padding='SAME')
        b18 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        UCL3 = tf.nn.bias_add(UCL3, b18)



        UCL4 = tf.add(UCL3, FCL5_from_pool3)

        ###########업샘플 3
        W19 = tf.Variable(tf.random_normal([16, 16, label.shape[3].value, label.shape[3].value], stddev=0.01),
                          dtype=np.float32)
        UCL5 = tf.nn.conv2d_transpose(UCL4, W19, output_shape=[self.input_size, label.shape[1].value, label.shape[2].value,
                                                               label.shape[3].value], strides=[1, 8, 8, 1],
                                      padding='SAME')
        b19 = tf.Variable(tf.random_normal([label.shape[3].value]), dtype=np.float32)
        UCL5 = tf.nn.bias_add(UCL5, b19)

        prediction = tf.argmax(UCL5, dimension=3)
        prediction = tf.expand_dims(prediction, dim=3)

        return prediction, UCL5

    def train(self, input, label, keep_prob, is_training):
        pred, logit_ = self.build(input, label, keep_prob, is_training)

        logit = tf.reshape(logit_, [-1, label.shape[3].value])
        label = tf.reshape(label, [-1, label.shape[3].value])

        loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=label)))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        return train_op, pred, loss, logit_

    def run(self, max_iter, already_done):
        ######already_done = 이미 에폭 돌아간 횟수########


        img_loader = data_load.ImgLoader()
        img_loader.run('train') # train or val

        data_size = len(img_loader.class_path)
        batch_num = data_size//self.input_size


        # input_batch, label_batch = img_loader.nextbatch(self.input_size, iter)
        input_batch, label_batch = img_loader.nextbatch(self.input_size, 0)

        X = tf.placeholder(tf.float32, [None, input_batch.shape[1], input_batch.shape[2], input_batch.shape[3]])  # h * w * 3
        Y = tf.placeholder(tf.float32,
                           [None, label_batch.shape[1], label_batch.shape[2], label_batch.shape[3]])  # h * w * (class+1) <= 배경포함(+1)
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool)

        train_op, pred, loss, logit = self.train(X, Y, keep_prob, is_training)

        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())


        for epoch in range(max_iter) :
            for itr in range(batch_num):
                input_batch, label_batch = img_loader.nextbatch(self.input_size, itr)
                _, loss_, pred_ = sess.run([train_op, loss, pred], feed_dict={X: input_batch, Y: label_batch, keep_prob: 0.5, is_training: True})
                print('iteration :', itr, '  loss :', loss_)


            if (epoch+1) % 2 == 0 :
                IOU_this_batch = util.intersection_over_union(pred_, label_batch)
                print('########################\n', 'intersection over union for this batch :', IOU_this_batch,
                      '\n########################')
                IOU_val = self.validation_IOU(sess, pred, X, Y, keep_prob, is_training)
                print('########################\n', 'intersection over union for validation set:', IOU_val,
                      '\n########################')

                model_dir = './model'+str(epoch+1+already_done)+'/model.ckpt'
                with open('loss.txt', 'a') as wf:
                    loss_info = '\nepoch: ' + '%7d'%(epoch+1+already_done) + '  batch loss: ' + '%7.6f'%loss_ + '  batch IOU: ' + '%7.6f'%IOU_this_batch + '  valiation IOU: ' + '%7.6f'%IOU_val
                    wf.write(loss_info)
                saver.save(sess, model_dir)
            else :
                model_dir = './model/model.ckpt'
                saver.save(sess, model_dir)


    def validation_IOU(self, sess, pred, X, Y, keep_prob, is_training):
        img_loader2 = data_load.ImgLoader('pascal')
        img_loader2.run('val')  # train or val

        batch_num= 10
        input_batch, label_batch = img_loader2.nextbatch(self.input_size, 0)

        IOU_sum=0

        for i in range(batch_num) :
            input_batch, label_batch = img_loader2.nextbatch(self.input_size, i)
            pred_=sess.run(pred, feed_dict={X: input_batch, Y: label_batch, keep_prob: 1.0, is_training: False})
            IOU = util.intersection_over_union(pred_, label_batch)
            IOU_sum += IOU

        return IOU_sum/batch_num
