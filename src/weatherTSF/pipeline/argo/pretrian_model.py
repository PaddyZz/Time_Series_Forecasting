import pickle

def splitDataAndNormalization(inputs=False,df=None):
    if inputs is True and df is not None:
        pass
    else:
        with open('/models/lstm/df.pkl', 'rb') as f:
            df = pickle.load(f)
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]
    num_features = df.shape[1]
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std #validation_df
    test_df = (test_df - train_mean) / train_std

    if input is True:
        return train_df, val_df, test_df
    with open('/models/lstm/train.pkl', 'wb') as f:
        pickle.dump(train_df, f)
    with open('/models/lstm/val.pkl', 'wb') as f:
        pickle.dump(val_df, f)
    with open('/models/lstm/test.pkl', 'wb') as f:
        pickle.dump(test_df, f)
    return None
if __name__ == "__main__":

    splitDataAndNormalization()





    