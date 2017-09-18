import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
np.set_printoptions(threshold=np.inf, precision=2)
import tensorflow as tf


def cond_dropout(incoming, keep_prob, noise_shape=None, name="Dropout"):
    """ Dropout.

    Outputs the input element scaled up by `1 / keep_prob`. The scaling is so
    that the expected sum is unchanged.

    By default, each element is kept or dropped independently. If noise_shape
    is specified, it must be broadcastable to the shape of x, and only dimensions
    with noise_shape[i] == shape(x)[i] will make independent decisions. For
    example, if shape(x) = [k, l, m, n] and noise_shape = [k, 1, 1, n], each
    batch and channel component will be kept independently and each row and column
    will be kept or not kept together.

    Arguments:
        incoming : A `Tensor`. The incoming tensor.
        keep_prob : A float representing the probability that each element
            is kept.
        noise_shape : A 1-D Tensor of type int32, representing the shape for
            randomly generated keep/drop flags.
        name : A name for this layer (optional).

    References:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
        N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever & R. Salakhutdinov,
        (2014), Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.

    Links:
      [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf]
        (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

    """

    with tf.name_scope(name) as scope:

        inference = incoming

        def apply_dropout():

            # BATCH_SIZE = 32
            # TIME_STEPS = max_len
            # with tf.variable_scope('encoder') as scope:
            #     rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(128) for _ in range(3)])
            #     print(rnn_cell.state_size)
            #     state = tf.zeros((np.int32(BATCH_SIZE), rnn_cell.state_size),  dtype=np.object)
            #     output = [None] * TIME_STEPS
            #     for t in reversed(range(TIME_STEPS)):
            #         y_t = tf.reshape(inference[:, t, :], (BATCH_SIZE, -1))
            #         print(y_t)
            #         output[t], state = rnn_cell(y_t, state)
            #         scope.reuse_variables()
            #     y = tf.pack(output, 1)
            if type(inference) in [list, np.array]:
                for x in inference:
                    x = tf.nn.dropout(x, keep_prob, noise_shape)
                return inference
            else:
                return tf.nn.dropout(inference, keep_prob, noise_shape)

        is_training = tflearn.get_training_mode()
        inference = tf.cond(is_training, apply_dropout, lambda: inference)

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

df = pd.read_csv("Activity_April.csv")

df = df.drop(['(case) variant',
         '(case) variant-index', 'concept:name', 'lifecycle:transition', '(case) creator', 'Variant'], axis = 1)


to_numbers = dict(zip(df['Activity'].unique(), range(1, len(df['Activity'].unique())+1)))
df['Activity_Index'] = df['Activity']
df['Activity_Index'] = df['Activity_Index'] .replace(to_numbers)

df['Activity_Time'] = 0.0
df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
df['Activity_Time'] = df['Complete Timestamp'] - df['Start Timestamp']
df['Activity_Time'] = df['Activity_Time'].astype('timedelta64[h]')
df['Start Timestamp'] = df['Start Timestamp'].values.view('<i8')/10**9
df['Previous_Activity'] = df.groupby('Case ID')['Activity_Index'].shift(1).fillna(11.0).apply(np.array)
df['trigram_Activity'] = df.groupby('Case ID')['Activity_Index'].shift(2).fillna(11.0).apply(np.array)
df['Previous'] = df.groupby('Case ID')['Activity'].shift(1).fillna("None").apply(np.array)
df['XorLoop'] = 0
df['Xor'] = 0
df['And'] = 0
df['Seq'] = 0
df.loc[(df['Activity']=='Exam') & (df['Previous']=='Vitals'), ['Seq']] = 1
df.loc[df['Activity']=='Pharmacy', ['XorLoop']] = 1
df.loc[df['Activity']=='Exam', ['XorLoop']] = 1
df.loc[df['Activity']=='BloodDraw', ['XorLoop']] = 1
df.loc[(df['Activity']=='BloodDraw') & (df['Previous']=='Arrival'), ['Seq']] = 1
df.loc[(df['Activity']=='Infusion') & (df['Previous']=='Pharmacy'), ['And']] = 1
df.loc[(df['Activity']=='Pharmacy') & (df['Previous']=='Infusion'), ['And']] = 1
df.loc[(df['Activity']=='Exam') & (df['Previous']=='Arrival'), ['And']] = 1
df.loc[(df['Activity']=='Arrival') & (df['Previous']=='Exam'), ['And']] = 1
#df['Elapsed_Time'] = None
df['Number_of_previous_activities'] = None
#elapsed = {}
numbers = {}
for i,row in df.iterrows():
    if not numbers.get(row['Case ID'],False):
        #elapsed[row['Case ID']]=0
        numbers[row['Case ID']]=0

    #elapsed[row['Case ID']]+=row['Activity_Time']
    #df.set_value(i,'Elapsed_Time',elapsed[row['Case ID']])
    df.set_value(i,'Number_of_previous_activities',numbers[row['Case ID']])
    numbers[row['Case ID']]+=1
Case_IDs = pd.DataFrame({'count': df.groupby(['Case ID']).size()}).reset_index()
print(df)
df_y = df[['Case ID', 'Activity_Index', 'Start Timestamp']]
df_y['Activity_Index'] = df_y.groupby('Case ID')['Activity_Index'].shift(-1).fillna(7.0).apply(np.array)
df_y = df_y.sort_values(['Case ID','Start Timestamp']).drop(['Start Timestamp'], axis=1).groupby('Case ID').apply(np.array)
# max_len = 0
max_len = 6
df_y_new = []
for row in df_y:
    temp = []
    for col in row:
        temp += list(col[1:])
        # if len(temp) > max_len: break
    # if len(temp) > max_len : max_len= len(temp)
    if len(temp) > max_len or len(temp) < 2 : continue
    # if len(temp) < 2 : continue
    # print(len(temp))
    df_y_new += [temp, ]
    # df_y_new += [temp[-1]]

df_y = pad_sequences(df_y_new, maxlen=max_len)
df_y = np.reshape(df_y, (-1, max_len))

# df_y = np.reshape(df_y_new, (-1, 1))

# df = df[['Case ID', 'Activity_Index']]
# df = df.groupby('Case ID').apply(np.array)
# max_len = 0
# df_new = []
# for row in df:
#     temp = []
#     for col in row:
#         temp += list(col[1:])
#     if len(temp) > max_len: max_len = len(temp)
#     df_new += [temp, ]
#     # df_y_new += [temp[-1]]
# df = pad_sequences(df_new, maxlen=max_len, value=6)
# df = np.reshape(df, (-1, 18))

# df = df.drop(['Activity', 'Complete Timestamp', 'Variant index'], axis = 1)
# df = df.groupby('Case ID').apply(np.array)
# pad_seq = [0.0, 0.0, 0.0]
# df_new = []
# for row in df:
#     temp = []
#     for col in row:
#         temp += [list(col[1:]),]
#     for i in range(len(temp), max_len):
#         temp += [pad_seq, ]
#     df_new += [temp, ]
# df = np.array(list(df_new), dtype=np.float32)

df = df.join(pd.get_dummies(df['Activity'], prefix='Activity'))\
    .join(pd.get_dummies(df['Activity_Time'], prefix='Activity_Time'))\
    .join(pd.get_dummies(df['Previous_Activity'], prefix='Previous_Activity'))\
    .join(pd.get_dummies(df['trigram_Activity'], prefix='trigram_Activity'))\
    .join(pd.get_dummies(df['Elapsed_Time'], prefix='Elapsed_Time'))

df['Padding_Activity'] = 0
df = df.drop(['Activity', 'Complete Timestamp', 'Variant index', 'Activity_Index', \
              'Activity_Time', 'Previous_Activity','Previous', 'trigram_Activity', 'Elapsed_Time'], axis = 1)
pad_size = len(df.columns)
df = df.sort_values(['Case ID', 'Start Timestamp']).groupby('Case ID').apply(np.array)
pad_seq = [0.0] * (pad_size - 2) + [1.0]
df_new = []
for row in df:
    temp = []
    for col in row:
        temp += [list(col[1:]), ]
    if len(temp) > max_len or len(temp) < 2: continue
    # if len(temp) < 2:continue
    for i in range(len(temp), max_len):
        temp += [pad_seq, ]
    df_new += [temp, ]

df = np.zeros((len(df_new), len(df_new[0]), len(df_new[0][0])), dtype=np.float32)
for i in range(len(df_new)):
    for j in range(len(df_new[0])):
        for k in range(len(df_new[0][0])):
            df[i][j][k] = df_new[i][j][k]

trainX, testX, trainY, testY = train_test_split(df, df_y, test_size=0.24)
# print(testY[:100])
trainY = to_categorical(trainY.ravel(), nb_classes=8)
testY = to_categorical(testY.ravel(), nb_classes=8)
print(testY[:100])
learning_rates = [0.001]
epochs=[100]
for lr in learning_rates:
    for epoch in epochs:

        net = tflearn.input_data(shape=[None] + list(trainX.shape)[1:])
        # net = tflearn.input_data(shape=[None] + [18, 21])
        net = tflearn.layers.normalization.batch_normalization(net)
        # net = tflearn.lstm(net, 128, return_seq=True, dynamic=True, activation='relu')
        # cell = tflearn.layers.BasicLSTMCell(128)
        # cell.add_update()
        net = tflearn.lstm(net, 128, activation='relu', dynamic=True)
        net = cond_dropout(net, 0.8)
        # net = tflearn.fully_connected(net, 18, activation='linear')

        # net = tflearn.embedding(net, input_dim=10000, output_dim=128)
        # net = tflearn.lstm(net, 128, dropout=0.8)
        # net = tflearn.fully_connected(net, 18, activation='linear')
        # W = tf.Variable(tf.random_normal([784, 256]), name="W")
        # T = tflearn.helpers.add_weights_regularizer(W, 'L2', weight_decay=0.001)
        net = tflearn.fully_connected(net, 8, activation='softmax')
        W=net.W
        save_net = net
        net = tflearn.regression(net, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', shuffle_batches=False)
                                 # ,to_one_hot=True, n_classes=8)

        model = tflearn.DNN(net, tensorboard_verbose=3)
        model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
                  batch_size=6, n_epoch=epoch)

        print("Regular:" + str(model.evaluate(testX, testY)))
        predY = model.predict(testX)
        predYnorm = np.zeros_like(predY)
        predYnorm[np.arange(len(predY)), predY.argmax(1)] = 1
        print(predYnorm[:100])

# new_net = tflearn.regression(save_net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy',
#                          shuffle_batches=False)
#
# W_new = model.get_weights(W)
# W_new[:,0] = [0]*len(W_new)
# model.set_weights(W, W_new)
# # model.save("my_model.tflearn")
# # model = tflearn.DNN(new_net).load("temp.model")
# #
# print("Zeros:" + str(model.evaluate(testX, testY)))
#
# model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
#           batch_size=6, n_epoch=1)
#
# print("One more Iter:" + str(model.evaluate(testX, testY)))
# predY = model.predict(testX)
# predYnorm = np.zeros_like(predY)
# predYnorm[np.arange(len(predY)), predY.argmax(1)] = 1
# print(predYnorm[:100])


# predY = to_categorical(predY, nb_classes=7)
# print(predY[:100])
# cm = confusion_matrix(testY,predY)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cm, classes=range(7),
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cm, classes=range(7), normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()
#
