import tensorflow as tf
import os
import glob
import pickle
import minio
from argo.compile_and_fit import compile_and_fit
from argo.data_windowing import WindowGenerator

 
minio_client = minio(
        "<your-minio-cluster-ip>:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
minio_bucket = "mlpipeline"

def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
		 
    assert os.path.isdir(local_path)

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(
                local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(
                minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(
                os.sep, "/")  # Replace \ with / on Windows
            minio_client.fput_object(bucket_name, remote_path, local_file)
def model_train():
    with open('/models/lstm/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    with open('/models/lstm/val.pkl', 'rb') as f:
        val_df = pickle.load(f)
    with open('/models/lstm/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    multi_window = WindowGenerator(
    input_width=24, label_width=24, shift=24,
    train_df=train_df,val_df=val_df,test_df=test_df,
        label_columns=None)
    multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, 24*features].
    tf.keras.layers.Dense(24*14,
                        kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([24, 14])
    ])

    _ = compile_and_fit(multi_lstm_model, multi_window)  
    tf.saved_model.save(multi_lstm_model,'/models/lstm/model_dir')
    upload_local_directory_to_minio("/models/lstm/model_dir",minio_bucket,"/models/lstm/model_dir/")      
if __name__=="__main__":
    model_train()