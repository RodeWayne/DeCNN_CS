import numpy as np
import tensorflow as tf
from DeCNN import train_data_prepare as tdp

batch_size = 128
test_batch_size = 128
training_epochs = 1000
display_step = 1
KEEP_PROB = 1.0
mr = 0.01
mr_str = str(mr).replace(".", "_")
#print("mr_str:", mr_str)

CS_input_1D_dim = tdp.crops_heigth * tdp.crops_width * 1
CS_output_1D_dim = round(CS_input_1D_dim * mr)


# 生成高斯随机测量矩阵
def ge_csphi(ge_new=True):
    if ge_new:
        CS_Phi = np.random.randn(CS_input_1D_dim, CS_output_1D_dim)  # 保证用的是同一个Phi
        u, s, vh = np.linalg.svd(CS_Phi)
        print("u.shape:", u.shape)
        CS_Phi = u[0:CS_input_1D_dim, 0:CS_output_1D_dim] #将测量矩阵正交化
        print("CS_Phi.shape:", CS_Phi.shape)
        np.save("./CS_Phi_mr_%s.npy" % mr_str, CS_Phi)
    else:
        CS_Phi = np.load("./CS_Phi_mr_%s.npy" % mr_str)
    return CS_Phi


def deConv_op_1(input_op, name, kh, kw, output_shape, n_out, dh, dw, p, myDeconvWeights):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        '''
        kernel_init_val = tf.constant(myDeconvWeights, shape=[kh, kw, n_in, n_out],
                                      dtype=tf.float32)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        '''
        kernel = tf.get_variable(scope+"w", dtype=tf.float32,
                                 initializer=myDeconvWeights)

        deConv = tf.nn.conv2d_transpose(input_op, kernel, output_shape=output_shape,
                                        strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.01, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(deConv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation

def deConv_op_2(input_op, name, kh, kw, output_shape, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", dtype=tf.float32, shape=[kh, kw, n_out, n_in],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        deConv = tf.nn.conv2d_transpose(input_op, kernel, output_shape=output_shape,
                                        strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.01, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(deConv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def deConv_op_3(input_op, name, kh, kw, output_shape, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", dtype=tf.float32, shape=[kh, kw, n_out, n_in],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        deConv = tf.nn.conv2d_transpose(input_op, kernel, output_shape=output_shape,
                                        strides=[1, dh, dw, 1], padding='VALID')
        bias_init_val = tf.constant(0.01, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(deConv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def Conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", dtype=tf.float32, shape=[kh, kw, n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        Conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.01, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(Conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def Conv_op_2(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", dtype=tf.float32, shape=[kh, kw, n_in, n_out],
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())

        Conv = tf.nn.conv2d(input_op, kernel, strides=[1, dh, dw, 1], padding='VALID')
        bias_init_val = tf.constant(0.01, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(Conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op_relu(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w", shape=[n_in, n_out],
                                 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out],
                                         dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation

'''
def get_DeconvWeights(initial_train_data=trainImgArrayList):
    print("Initialize Dec_weights")
    DecWL = []
    autoencoder1 = mcd2.get_autoencoder(MymodelName="autoed1", Mywkernel=[7, 5], MyStrides=[1, 2, 1, 1],
                                        MyInput_height=30, MyInput_weight=30, MyInput_channel=1,
                                        MyOutput_channel=48, feedtype=2, init_op=True)  # 更好的方法时自适应的决定feedtype
    print("get Dec_weights[0]:")
    mcd2.train(this_autoencoder=autoencoder1, input_is_initial=True,
              train_data=initial_train_data)
    autoencoder1.Save_model("autoed1")
    DecW1 = autoencoder1.getDeconvWeights()
    DecWL.append(DecW1)
    hidden1 = mcd2.generate_total_hidden(data=initial_train_data, input_is_initial=True,
                                         this_autoencoder=autoencoder1)
    # 此时的hidden1是2维的，所以auoencoder2的feedtype=2
    autoencoder2 = mcd2.get_autoencoder(MymodelName="autoed2", Mywkernel=[3, 3], MyStrides=[1, 1, 1, 1],
                                        MyInput_height=15, MyInput_weight=30, MyInput_channel=48,
                                        MyOutput_channel=80, feedtype=2, init_op=True)
    print("get Dec_weights[1]:")
    mcd2.train(this_autoencoder=autoencoder2, input_is_initial=False, train_data=hidden1)
    autoencoder2.Save_model("autoed2")
    DecW2 = autoencoder2.getDeconvWeights()
    DecWL.append(DecW2)
    hidden2 = mcd2.generate_total_hidden(data=hidden1, input_is_initial=False, this_autoencoder=autoencoder2)
    autoencoder3 = mcd2.get_autoencoder(MymodelName="autoed3", Mywkernel=[5, 5], MyStrides=[1, 1, 2, 1],
                                        MyInput_height=15, MyInput_weight=30, MyInput_channel=80,
                                        MyOutput_channel=150, feedtype=2, init_op=True)
    print("get Dec_weights[2]:")
    mcd2.train(this_autoencoder=autoencoder3, input_is_initial=False, train_data=hidden2)
    autoencoder3.Save_model("autoed3")
    DecW3 = autoencoder3.getDeconvWeights()
    DecWL.append(DecW3)
    hidden3 = mcd2.generate_total_hidden(data=hidden2, input_is_initial=False, this_autoencoder=autoencoder3)
    autoencoder4 = mcd2.get_autoencoder(MymodelName="autoed4", Mywkernel=[3, 3], MyStrides=[1, 1, 1, 1],
                                        MyInput_height=15, MyInput_weight=15, MyInput_channel=150,
                                        MyOutput_channel=64, feedtype=2, init_op=True)
    print("get Dec_weights[3]:")
    mcd2.train(this_autoencoder=autoencoder4, input_is_initial=False, train_data=hidden3)
    autoencoder4.Save_model("autoed4")
    DecW4 = autoencoder4.getDeconvWeights()
    DecWL.append(DecW4)
    hidden4 = mcd2.generate_total_hidden(data=hidden3, input_is_initial=False, this_autoencoder=autoencoder4)
    autoencoder5 = mcd2.get_autoencoder(MymodelName="autoed5", Mywkernel=[3, 3], MyStrides=[1, 1, 1, 1],
                                        MyInput_height=15, MyInput_weight=15, MyInput_channel=64,
                                        MyOutput_channel=32, feedtype=2, init_op=True)
    print("get Dec_weights[4]:")
    mcd2.train(this_autoencoder=autoencoder5, input_is_initial=False, train_data=hidden4)
    autoencoder5.Save_model("autoed5")
    DecW5 = autoencoder5.getDeconvWeights()
    DecWL.append(DecW5)
    hidden5 = mcd2.generate_total_hidden(data=hidden4, input_is_initial=False, this_autoencoder=autoencoder5)
    autoencoder6 = mcd2.get_autoencoder(MymodelName="autoed6", Mywkernel=[3, 3], MyStrides=[1, 1, 1, 1],
                                        MyInput_height=15, MyInput_weight=15, MyInput_channel=32,
                                        MyOutput_channel=1, feedtype=2, init_op=True)
    print("get Dec_weights[5]:")
    mcd2.train(this_autoencoder=autoencoder6, input_is_initial=False, train_data=hidden5)
    autoencoder6.Save_model("autoed6")
    DecW6 = autoencoder6.getDeconvWeights()
    DecWL.append(DecW6)
    print("len(DecWL):", len(DecWL))
    print("Dec_weights have been initialized.")
    return DecWL
'''


def inference_op(input_op, keep_prob):
    p = []
    fc1 = fc_op_relu(input_op, name="fc1", n_out=300, p=p)
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name="fc1_drop")
    fc2 = fc_op_relu(fc1_drop, name="fc2", n_out=450, p=p)
    fc2_drop = tf.nn.dropout(fc2, keep_prob, name="fc2_drop")
    fc3 = fc_op_relu(fc2_drop, name="fc3", n_out=600, p=p)
    fc3_drop = tf.nn.dropout(fc3, 0.8, name="fc3_drop")
    fc4 = fc_op_relu(fc3_drop, name="fc4", n_out=750, p=p)
    fc4_drop = tf.nn.dropout(fc4, 0.9, name="fc1_drop")
    fc5 = fc_op_relu(fc4_drop, name="fc5", n_out=900, p=p)
    resh = tf.reshape(fc5, [batch_size, 30, 30, 1], name="resh")
    conv1 = Conv_op(input_op=resh, name="conv1", kh=3, kw=3, n_out=1, dh=1, dw=1, p=p)

    return conv1


def inference_op2(input_op, keep_prob, myflag, initial_w=1):
    p = []
    fc1 = fc_op_relu(input_op, name="fc1", n_out=16, p=p)
    fc1_drop = tf.nn.dropout(fc1, keep_prob, name="fc1_drop")
    #print("In inference_op2, myflag.batch_size:", myflag.batch_size)
    resh1 = tf.reshape(fc1_drop, [-1, 4, 4, 1], name="resh1")
    print("resh1.shape:", resh1.shape)

    if initial_w == 1:
        '''
        DecWL = get_DeconvWeights()
        deConv1_1 = deConv_op_1(resh1, name="deConv1_1", kh=3, kw=3, output_shape=(batch_size, 15, 15, 32), n_out=32,
                                dh=1, dw=1, p=p, myDeconvWeights=DecWL[-1])
        deConv1_2 = deConv_op_1(deConv1_1, name="deConv1_2", kh=3, kw=3, output_shape=(batch_size, 15, 15, 64), n_out=64,
                                dh=1, dw=1, p=p, myDeconvWeights=DecWL[-2])
        deConv1_3 = deConv_op_1(deConv1_2, name="deConv1_3", kh=3, kw=3, output_shape=(batch_size, 15, 15, 150), n_out=150,
                                dh=1, dw=1, p=p, myDeconvWeights=DecWL[-3])
        deConv2_1 = deConv_op_1(deConv1_3, name="deConv2_1", kh=5, kw=5, output_shape=(batch_size, 15, 30, 80), n_out=80,
                                dh=1, dw=2, p=p, myDeconvWeights=DecWL[-4])
        deConv2_2 = deConv_op_1(deConv2_1, name="deConv1_2", kh=3, kw=3, output_shape=(batch_size, 15, 30, 48), n_out=48,
                                dh=1, dw=1, p=p, myDeconvWeights=DecWL[-5])
        deConv2_3 = deConv_op_1(deConv2_2, name="deConv1_2", kh=7, kw=5, output_shape=(batch_size, 30, 30, 1), n_out=1,
                                dh=2, dw=1, p=p, myDeconvWeights=DecWL[-6])
        '''
        pass

    else:
        '''
         # MR:0.25, 289-289
        deConv1 = deConv_op_2(resh1, name="deConv1", kh=3, kw=5, output_shape=(myflag.batch_size, 17, 34, 64),
                              n_out=64, dh=1, dw=2, p=p)
        conv1 = Conv_op(deConv1, name="conv1", kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
        conv2 = Conv_op(conv1, name="conv2", kh=3, kw=3, n_out=1, dh=1, dw=1, p=p)
        deConv2 = deConv_op_2(conv2, name="deConv2", kh=5, kw=3, output_shape=(myflag.batch_size, 34, 34, 64),
                              n_out=64, dh=2, dw=1, p=p)
        conv3 = Conv_op(deConv2, name="conv3", kh=3, kw=3, n_out=32, dh=1, dw=1, p=p)
        conv4 = Conv_op(conv3, name="conv4", kh=3, kw=3, n_out=1, dh=1, dw=1, p=p)

        '''
        '''
        #mr:0_1, 116-121
        deConv1 = deConv_op_2(resh1, name="deConv1", kh=3, kw=5, output_shape=(myflag.batch_size, 11, 22, 64),
                              n_out=64, dh=1, dw=2, p=p)
        #conv2 = Conv_op(deConv1, name="conv2", kh=3, kw=3, n_out=1, dh=1, dw=1, p=p)
        #conv2_drop = tf.nn.dropout(conv2, keep_prob=keep_prob, name="conv2_drop")
        deConv2 = deConv_op_2(deConv1, name="deConv2", kh=5, kw=3, output_shape=(myflag.batch_size, 22, 22, 32),
                              n_out=32, dh=2, dw=1, p=p)
        deConv3 = deConv_op_2(deConv2, name="deConv3", kh=3, kw=5, output_shape=(myflag.batch_size, 22, 44, 1),
                              n_out=1, dh=1, dw=2, p=p)
        deConv4 = deConv_op_2(deConv3, name="deConv4", kh=5, kw=3, output_shape=(myflag.batch_size, 44, 44, 64),
                              n_out=64, dh=2, dw=1, p=p)
        conv1 = Conv_op_2(deConv4, name="conv1", kh=7, kw=7, n_out=32, dh=1, dw=1, p=p)
        conv2 = Conv_op_2(conv1, name="conv2", kh=5, kw=5, n_out=1, dh=1, dw=1, p=p)
        '''
        '''
        # MR:0.04, 46-49
        deConv1 = deConv_op_2(resh1, name="deConv1", kh=3, kw=5, output_shape=(myflag.batch_size, 7, 14, 64),
                              n_out=64, dh=1, dw=2, p=p)
        deConv2 = deConv_op_2(deConv1, name="deConv2", kh=5, kw=3, output_shape=(myflag.batch_size, 14, 14, 32),
                              n_out=32, dh=2, dw=1, p=p)
        deConv3 = deConv_op_2(deConv2, name="deConv3", kh=3, kw=5, output_shape=(myflag.batch_size, 14, 42, 1),
                              n_out=1, dh=1, dw=3, p=p)
        deConv4 = deConv_op_2(deConv3, name="deConv4", kh=5, kw=3, output_shape=(myflag.batch_size, 42, 42, 64),
                              n_out=64, dh=3, dw=1, p=p)
        conv1 = Conv_op_2(deConv4, name="conv1", kh=5, kw=5, n_out=32, dh=1, dw=1, p=p)
        conv2 = Conv_op_2(conv1, name="conv2", kh=5, kw=5, n_out=1, dh=1, dw=1, p=p)
        '''

        # MR: 0.01, 12-16
        deConv1 = deConv_op_2(resh1, name="deConv1", kh=3, kw=3, output_shape=(myflag.batch_size, 4, 12, 64),
                              n_out=64, dh=1, dw=3, p=p)
        deConv2 = deConv_op_2(deConv1, name="deConv2", kh=3, kw=3, output_shape=(myflag.batch_size, 12, 12, 32),
                              n_out=32, dh=3, dw=1, p=p)
        deConv3 = deConv_op_2(deConv2, name="deConv3", kh=3, kw=3, output_shape=(myflag.batch_size, 24, 24, 1),
                              n_out=1, dh=2, dw=2, p=p)
        deConv4 = deConv_op_2(deConv3, name="deConv4", kh=3, kw=3, output_shape=(myflag.batch_size, 48, 48, 32),
                              n_out=32, dh=2, dw=2, p=p)
        conv1 = Conv_op_2(deConv4, name="conv1", kh=8, kw=8, n_out=16, dh=1, dw=1, p=p)
        conv2 = Conv_op_2(conv1, name="conv2", kh=8, kw=8, n_out=1, dh=1, dw=1, p=p)


    return conv2

'''
def compressed_op(input_op, myflag):
    p=[]
    conv1_1 = Conv_op(input_op, name="conv1", kh=3, kw=3, n_out=1, dh=2, dw=1, p=p)
    conv1_2 = Conv_op(conv1_1, name="conv2", kh=3, kw=3, n_out=1, dh=1, dw=2, p=p)
    conv1_3 = Conv_op(conv1_2, name="conv3", kh=3, kw=3, n_out=1, dh=1, dw=1, p=p)
    resh = tf.reshape(conv1_3, [myflag.batch_size, CS_output_1D_dim])
    return resh

'''


class DeconvNN_CS(object):
    def __init__(self, myflag, optimizer=tf.train.AdamOptimizer(), Input_height=tdp.crops_heigth,
                 Input_weight=tdp.crops_width, Input_channel=1, init_op=True):
        self.input_1D_dim = Input_height * Input_weight * Input_channel
        self.original_images_2D = tf.placeholder(tf.float32, shape=[None, Input_height, Input_weight, Input_channel])
        self.original_images_1D = tf.reshape(self.original_images_2D, shape=[-1, self.input_1D_dim])
        self.CS_Phi = tf.placeholder(tf.float32, [self.input_1D_dim, CS_output_1D_dim])
        self.compressed_images_1D = tf.matmul(self.original_images_1D, self.CS_Phi)
        self.keep_prob = KEEP_PROB
        self.recovered_images_2D = inference_op2(self.compressed_images_1D, self.keep_prob, myflag=myflag, initial_w=2)
        self.recovered_images_1D = tf.reshape(self.recovered_images_2D, [myflag.batch_size, self.input_1D_dim])
        self.cost = (1.0 / (255.0 * 255.0)) * tf.div(
            tf.reduce_sum(tf.pow(tf.subtract(self.recovered_images_1D, self.original_images_1D), 2.0)),
            self.input_1D_dim)
        self.optimizer = optimizer.minimize(self.cost)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if init_op:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            self.saver.restore(self.sess, "./sess_v1/DeconvNN_CS1_mr_%s_e1000.ckpt" % mr_str)
            print("DeconvNN_CS1_mr_%s_e1000 model restored." % mr_str)

    def Save_model(self):
        save_path = self.saver.save(self.sess, "./sess_v1/DeconvNN_CS1_mr_%s_e1000.ckpt" % mr_str)
        print("model saved in file: %s." % save_path)

    def partial_fit(self, X_2D, CS_Phi):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.original_images_2D: X_2D, self.CS_Phi: CS_Phi})
        return cost

    def calc_cost(self, X_2D, CS_Phi):
        cost = self.sess.run(self.cost,
                             feed_dict={self.original_images_2D: X_2D, self.CS_Phi: CS_Phi})
        return cost

    def getReconstruct(self, X_2D, CS_Phi):
        recovered = self.sess.run(self.recovered_images_2D,
                                  feed_dict={self.original_images_2D: X_2D, self.CS_Phi: CS_Phi})
        return recovered


class myflag(object):
    def __init__(self, model="train"):
        self.model = model
        self.batch_size = 128

    def check_batch_size(self):
        if self.model == "test":
            self.batch_size = 1
        elif self.model == 'train':
            self.batch_size = 128


def get_random_block_from_train_data(data, data_batch_size):
    start_index = np.random.randint(0, len(data) - data_batch_size)
    return data[start_index:(start_index + data_batch_size)]


def get_block_from_data(data, data_batch_size, j):
    start_index = j * data_batch_size
    return data[start_index:(start_index + data_batch_size)]


def De_train(myDeconvNN_CS, with_new_csphi, myflag):
    train_data_dirs = "./Training_Data_L/Train/"
    trainImgList = tdp.eachFile(train_data_dirs)
    trainImgArrayList_ori = tdp.crop_img(trainImgList)
    trainImgArrayList = tdp.up_dim(trainImgArrayList_ori)

    myflag.model = "train"
    myflag.check_batch_size()
    CS_Phi = ge_csphi(with_new_csphi)
    # train中平均每个batch的总损失
    batch_num = int(len(trainImgArrayList) / myflag.batch_size)
    for epoch in range(training_epochs):
        total_train_cost = 0.
        for i in range(batch_num):
            image_batch = get_block_from_data(trainImgArrayList, myflag.batch_size, i)
            train_cost_batch = myDeconvNN_CS.partial_fit(X_2D=image_batch, CS_Phi=CS_Phi)
            total_train_cost += train_cost_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "De_one_train_cost=",
                  "{:.9f}".format(total_train_cost / (myflag.batch_size * batch_num)))
            '''
            print("Epoch:", '%04d' % (epoch + 1), "De_train_batch_cost=",
                  "{:.9f}".format((total_train_cost / batch_num)))
            '''
        if (epoch+1) % 10 == 0:
            De_test(myDeconvNN_CS=myDeconvNN_CS, with_new_csphi=False, myflag=myflag)

    print("In train phase, batch_size:", myflag.batch_size)
    print("De_one_train_cost=", "{:.9f}".format(total_train_cost / (myflag.batch_size * batch_num)))


def De_test(myDeconvNN_CS, with_new_csphi, myflag):

    test_data_list = "./Training_Data_L/validation1/"
    testImgList = tdp.eachFile(test_data_list)
    testImgArrayList_ori = tdp.crop_img(testImgList, crop_strides=tdp.crops_width)
    testImgArrayList = tdp.up_dim(testImgArrayList_ori)

    #myflag.model = "test"
    #myflag.check_batch_size()
    #print("In test, myflag.batch_size:", myflag.batch_size)
    CS_Phi = ge_csphi(with_new_csphi)
    # test数据集的总损失
    total_test_cost = 0.
    batch_num = int(len(testImgArrayList) / myflag.batch_size)
    #print("testImgArrayList[0].shape:", testImgArrayList[0].shape)
    for i in range(batch_num):
        img_batch = get_block_from_data(testImgArrayList, myflag.batch_size, i)
        test_cost_batch = myDeconvNN_CS.calc_cost(X_2D=img_batch, CS_Phi=CS_Phi)
        total_test_cost += test_cost_batch
    print("In test phase, batch_size:", myflag.batch_size)
    #print("De_test_batch_cost=", "{:.9f}".format(total_test_cost / batch_num))
    print("De_one_test_cost=", "{:.9f}".format(total_test_cost / (batch_num * myflag.batch_size)))
def De_ops(init_op=True):
    FLAG = myflag()
    DeCNN_CS = DeconvNN_CS(myflag=FLAG, optimizer=tf.train.AdamOptimizer(learning_rate=0.0005), init_op=init_op)
    if init_op == True:
        De_train(myDeconvNN_CS=DeCNN_CS, with_new_csphi=False, myflag=FLAG)
        DeCNN_CS.Save_model()
        De_test(myDeconvNN_CS=DeCNN_CS, with_new_csphi=False, myflag=FLAG)
    else:
        #De_test(myDeconvNN_CS=DeCNN_CS, with_new_csphi=False, myflag=FLAG)
        pass
    return DeCNN_CS


if __name__ == '__main__':
    De_ops(init_op=True)



