model:
	wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz
mm:
	wget https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_075_224/classification/5/default/1?lite-format=tflite -O mobilenet_v3_small_075_224.tflite
