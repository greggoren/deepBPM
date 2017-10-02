import pandas as pd
import numpy as np



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


def preprocess_data_set(df,features):
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
            ''
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
