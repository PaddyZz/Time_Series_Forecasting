from kserve import KServeClient
import requests
import tensorflow as tf 
from argo.data_ingestion import getDataset
from argo.pretrian_model import splitDataAndNormalization
from argo.data_windowing import WindowGenerator

def model_evaluating():
	df = getDataset(inputs=True)        
	train_df,val_df,test_df,_ = splitDataAndNormalization(inputs=True,df=df)
	wide_window = WindowGenerator(
	        input_width=24, label_width=24, shift=24,
	        train_df=train_df,val_df=val_df,test_df=test_df,
	        label_columns=None)
	inputs, _=wide_window.example
	
	KServe = KServeClient()
	
	isvc_resp = KServe.get("weather-prediction-kserve-inference-service-v1", namespace="kubeflow-user-example-com")
	isvc_url = isvc_resp['status']['address']['url']
	inference_input = {
	  'instances': inputs.numpy().tolist()
	}
	
	response = requests.post(isvc_url, json=inference_input)
	predictions = response.json()['predictions']
	pred_tensor = tf.convert_to_tensor(predictions)
	wide_window.plot(pred_tensor=pred_tensor)

if __name__ == "__main__":
    model_evaluating()