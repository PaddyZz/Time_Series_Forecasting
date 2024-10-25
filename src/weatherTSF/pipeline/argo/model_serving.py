import kfp
import kfp.components as components
from kubernetes import client 
from kserve import KServeClient
from kserve import constants
from kserve import utils
from kserve import V1beta1InferenceService
from kserve import V1beta1InferenceServiceSpec
from kserve import V1beta1PredictorSpec
from kserve import V1beta1TFServingSpec
from kfp import dsl
import requests

def model_serving():
    """
    Create kserve instance
    """


    namespace = utils.get_default_target_namespace()


    name='weather-prediction-kserve-inference-service-v1'
    kserve_version='v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(api_version=api_version,
                                   kind=constants.KSERVE_KIND,
                                   metadata=client.V1ObjectMeta(
                                       name=name, namespace=namespace, annotations={'sidecar.istio.io/inject':'false'}),
                                   spec=V1beta1InferenceServiceSpec(
                                   predictor=V1beta1PredictorSpec(
                                       service_account_name="sa-minio-kserve", #see apply-minio-secret-to-kserve.yaml
                                       tensorflow=(V1beta1TFServingSpec(
                                           storage_uri="s3://mlpipeline/models/lstm/model_dir/"))))
    )

    KServe = KServeClient()
    KServe.create(isvc)
comp_model_serving = components.create_component_from_func(model_serving,base_image="public.ecr.aws/j1r0q0g6/notebooks/notebook-servers/jupyter-tensorflow-full:v1.5.0",
                                                           packages_to_install=['kserve==0.8.0.1'])   


@dsl.pipeline(
    name='weather-pred-pipeline',
    description='weather-prediction')
def weather_pred_pipeline():

    comp_model_serving()

if __name__ == "__main__":
    USERNAME="user@example.com"
    PASSWORD="12341234"
    NAMESPACE="kubeflow-user-example-com"
	HOST='http://istio-ingressgateway.istio-system.svc.cluster.local:80'
	session=requests.Session()
	response=session.get(HOST)
	headers={"Content-Type": "application/x-www-form-urlencoded",}
	data={"login": USERNAME, "password": PASSWORD}
	session.post(response.url, headers=headers, data=data)
	session_cookie=session.cookies.get_dict()["authservice_session"]
	client=kfp.Client(host=f"{HOST}/pipeline", cookies=f"authservice_session={session_cookie}",)
    client.create_run_from_pipeline_func(weather_pred_pipeline,arguments=None,experiment_name="weather-prediction-v1-exp") 
                                                        