import pandas as pd
import numpy as np
from itertools import  chain,combinations
from tflearn.data_utils import to_categorical, pad_sequences
import tflearn
from sklearn.model_selection import train_test_split
from copy import deepcopy
def preprocess_data_set(df_t,features):
    df=deepcopy(df_t)
    max_len=6
    df['Activity_Time'] = 0.0
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['Start Timestamp'] = pd.to_datetime(df['Start Timestamp'])
    df['Activity_Time'] = df['Complete Timestamp'] - df['Start Timestamp']
    df['Activity_Time'] = df['Activity_Time'].astype('timedelta64[h]')
    df['Start Timestamp'] = df['Start Timestamp'].values.view('<i8') / 10 ** 9
    df = df.join(pd.get_dummies(df['Activity'], prefix='Activity')) \
        .join(pd.get_dummies(df['Activity_Time'], prefix='Activity_Time'))
    for feature in features:
        if feature=='Previous_Activity':
            df['Previous_Activity'] = df.groupby('Case ID')['Activity_Index'].shift(1).fillna(11.0).apply(np.array)
            df = df.join(pd.get_dummies(df['Previous_Activity'], prefix='Previous_Activity'))
        if feature=='Trigram_Activity':
            df['Trigram_Activity'] = df.groupby('Case ID')['Activity_Index'].shift(2).fillna(11.0).apply(np.array)
            df = df.join(pd.get_dummies(df['Trigram_Activity'], prefix='Trigram_Activity'))
        if feature =='Number_of_previous_activities':
            df['Number_of_previous_activities'] = None
            # elapsed = {}
            numbers = {}
            for i, row in df.iterrows():
                if not numbers.get(row['Case ID'], False):
                    # elapsed[row['Case ID']]=0
                    numbers[row['Case ID']] = 0

                # elapsed[row['Case ID']]+=row['Activity_Time']
                # df.set_value(i,'Elapsed_Time',elapsed[row['Case ID']])
                df.set_value(i, 'Number_of_previous_activities', numbers[row['Case ID']])
                numbers[row['Case ID']] += 1
            df = df.join(pd.get_dummies(df['Number_of_previous_activities'], prefix='Number_of_previous_activities'))
        if feature=='Elapsed_Time':
            df['Elapsed_Time'] = None
            elapsed = {}
            for i, row in df.iterrows():
                if not numbers.get(row['Case ID'], False):
                    elapsed[row['Case ID']]=0

                elapsed[row['Case ID']]+=row['Activity_Time']
                df.set_value(i,'Elapsed_Time',elapsed[row['Case ID']])
            df = df.join(pd.get_dummies(df['Elapsed_Time'], prefix='Elapsed_Time'))
        if feature=='Tree':
            df['Previous'] = df.groupby('Case ID')['Activity'].shift(1).fillna("None").apply(np.array)
            df['XorLoop'] = 0
            df['Xor'] = 0
            df['And'] = 0
            df['Seq'] = 0
            df.loc[(df['Activity'] == 'Exam') & (df['Previous'] == 'Vitals'), ['Seq']] = 1
            df.loc[df['Activity'] == 'Pharmacy', ['XorLoop']] = 1
            df.loc[df['Activity'] == 'Exam', ['XorLoop']] = 1
            df.loc[df['Activity'] == 'BloodDraw', ['XorLoop']] = 1
            df.loc[(df['Activity'] == 'BloodDraw') & (df['Previous'] == 'Arrival'), ['Seq']] = 1
            df.loc[(df['Activity'] == 'Infusion') & (df['Previous'] == 'Pharmacy'), ['And']] = 1
            df.loc[(df['Activity'] == 'Pharmacy') & (df['Previous'] == 'Infusion'), ['And']] = 1
            df.loc[(df['Activity'] == 'Exam') & (df['Previous'] == 'Arrival'), ['And']] = 1
            df.loc[(df['Activity'] == 'Arrival') & (df['Previous'] == 'Exam'), ['And']] = 1
    feature_without_tree = [x for x in features if x!='Tree']
    drop_features = ['Activity', 'Complete Timestamp', 'Variant index', 'Activity_Index','Activity_Time']+feature_without_tree
    df = df.drop(drop_features,axis=1)
    df['Padding_Activity'] = 0
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
    return df

def create_all_feature_combinations():
    features = ['Tree','Elapsed_Time','Number_of_previous_activities','Previous_Activity','Trigram_Activity']
    feature_power_set = powerset(features)
    return feature_power_set

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def create_y(df):
    df_y = df[['Case ID', 'Activity_Index', 'Start Timestamp']]
    df_y['Activity_Index'] = df_y.groupby('Case ID')['Activity_Index'].shift(-1).fillna(7.0).apply(np.array)
    df_y = df_y.sort_values(['Case ID', 'Start Timestamp']).drop(['Start Timestamp'], axis=1).groupby('Case ID').apply(
        np.array)
    # max_len = 0
    max_len = 6
    df_y_new = []
    for row in df_y:
        temp = []
        for col in row:
            temp += list(col[1:])
            # if len(temp) > max_len: break
        # if len(temp) > max_len : max_len= len(temp)
        if len(temp) > max_len or len(temp) < 2: continue
        df_y_new += [temp, ]
        # df_y_new += [temp[-1]]

    df_y = pad_sequences(df_y_new, maxlen=max_len)
    df_y = np.reshape(df_y, (-1, max_len))
    return df_y

def create_experiment():
    combs=create_all_feature_combinations()
    df = pd.read_csv("Activity_April.csv")
    df = df.drop(['(case) variant',
                  '(case) variant-index', 'concept:name', 'lifecycle:transition', '(case) creator', 'Variant'], axis=1)
    to_numbers = dict(zip(df['Activity'].unique(), range(1, len(df['Activity'].unique()) + 1)))
    df['Activity_Index'] = df['Activity']
    df['Activity_Index'] = df['Activity_Index'].replace(to_numbers)
    df_y=create_y(df)
    results = open("res.txt", 'w')
    for features in combs:
        df1 = preprocess_data_set(df,features)
        trainX, testX, trainY, testY = train_test_split(df1, df_y, test_size=0.24)
        # print(testY[:100])
        trainY = to_categorical(trainY.ravel(), nb_classes=8)
        testY = to_categorical(testY.ravel(), nb_classes=8)
        print(testY[:100])
        learning_rates = [0.0001,0.001,0.01]
        epochs = [100]
        for lr in learning_rates:
            for epoch in epochs:
                net = tflearn.input_data(shape=[None] + list(trainX.shape)[1:])
                # net = tflearn.input_data(shape=[None] + [18, 21])
                net = tflearn.layers.normalization.batch_normalization(net)
                # net = tflearn.lstm(net, 128, return_seq=True, dynamic=True, activation='relu')
                # cell = tflearn.layers.BasicLSTMCell(128)
                # cell.add_update()
                net = tflearn.lstm(net, 128, activation='relu', dynamic=True)
                net = tflearn.dropout(net, 0.8)
                net = tflearn.fully_connected(net, 8, activation='softmax')
                net = tflearn.regression(net, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy',
                                         shuffle_batches=False)

                model = tflearn.DNN(net, tensorboard_verbose=3)
                model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
                          batch_size=6, n_epoch=epoch,snapshot_epoch=False)
                predY = model.predict(testX)
                predYnorm = np.zeros_like(predY)
                predYnorm[np.arange(len(predY)), predY.argmax(1)] = 1
                print(predYnorm[:100])
                combine = tuple(zip([j for i in predYnorm for j in i], [j for i in testY for j in i]))
                combine = [(x, y) for x, y in combine if y != 0]
                accuracy = sum(1 for x, y in combine if (x == y)) / float(len(combine))

                print("accuracy on test is fucking:", accuracy," lr= " ,lr,"features= ",features)
                try:
                    feat_for_print = "_".join(features)
                    results.write(feat_for_print+","+str(lr)+","+str(accuracy)+"\n")
                except:
                    print("problem in file writing look at log")


    results.close()

if __name__=="__main__":
    create_experiment()