#include "classifier.hpp"

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/shuffle_channel_layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
typedef caffe::Net<float>* caffe_net_ptr;

namespace caffe {
extern INSTANTIATE_CLASS(InputLayer);
extern INSTANTIATE_CLASS(SplitLayer);
// REGISTER_LAYER_CLASS(Input);
extern INSTANTIATE_CLASS(ConvolutionLayer);
//REGISTER_LAYER_CLASS(Convolution);
extern INSTANTIATE_CLASS(DepthwiseConvolutionLayer);
//REGISTER_LAYER_CLASS(DepthwiseConvolution);
extern INSTANTIATE_CLASS(BatchNormLayer);
//REGISTER_LAYER_CLASS(BatchNorm);
extern INSTANTIATE_CLASS(ScaleLayer);
//REGISTER_LAYER_CLASS(Scale);
extern INSTANTIATE_CLASS(BiasLayer);
extern INSTANTIATE_CLASS(ReLULayer);
extern INSTANTIATE_CLASS(ConcatLayer);
//REGISTER_LAYER_CLASS(Concat);
extern INSTANTIATE_CLASS(ShuffleChannelLayer);
//REGISTER_LAYER_CLASS(ShuffleChannel);
extern INSTANTIATE_CLASS(SliceLayer);
//REGISTER_LAYER_CLASS(Slice);
extern INSTANTIATE_CLASS(PoolingLayer);
//REGISTER_LAYER_CLASS(Pooling);
extern INSTANTIATE_CLASS(InnerProductLayer);
//REGISTER_LAYER_CLASS(InnerProduct);
extern INSTANTIATE_CLASS(SoftmaxLayer);
}


Classifier::Classifier(const std::string &model_file, const std::string &trained_file)
: mean_(127.5){
	// Load the network.
	//net_.reset(new caffe::Net<float>(model_file, caffe::TEST));
	net_ = new caffe::Net<float>(model_file, caffe::TEST);
	((caffe_net_ptr(net_)))->CopyTrainedLayersFrom(trained_file);
	CHECK_EQ((caffe_net_ptr(net_))->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ((caffe_net_ptr(net_))->num_outputs(), 1) << "Network should have exactly one output.";

	caffe::Blob<float> *input_layer = (caffe_net_ptr(net_))->input_blobs()[0];
	num_input_ = input_layer->num();
	input_gemotry_ = cv::Size(input_layer->width(), input_layer->height());
}

std::vector<float> Classifier::Predict(const std::vector<cv::Mat> &img_vec) {
	caffe::Blob<float> *input_layer = (caffe_net_ptr(net_))->input_blobs()[0];
	input_layer->Reshape(num_input_, 1, input_gemotry_.height, input_gemotry_.width);
	(caffe_net_ptr(net_))->Reshape();
	std::vector<cv::Mat> input_imgs;
	WrapInputLayer(&input_imgs);
	Preprocess(img_vec, &input_imgs);
	(caffe_net_ptr(net_))->Forward();
	// Copy the output layer to std::vector
	std::vector<float> liveness_score;
	caffe::Blob<float> *out = (caffe_net_ptr(net_))->output_blobs()[0];
	for (int i = 0; i < num_input_; ++i) {
		liveness_score.push_back(out->cpu_data()[i *2 + 1]);  // liveness score
	}
	return liveness_score;
}

void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_imgs) {
	caffe::Blob<float>* input_layer = (caffe_net_ptr(net_))->input_blobs()[0];
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels()*num_input_; ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_imgs->push_back(channel);
		input_data += width*height;
	}
}

void Classifier::Preprocess(const std::vector<cv::Mat>& img_vec,
	std::vector<cv::Mat> *input_imgs) {
	for (int i = 0; i < num_input_; ++i) {
		cv::Mat sample;
		if (img_vec[i].channels() != 1)
			cv::cvtColor(img_vec[i], sample, cv::COLOR_BGR2BGRA);
		else
			sample = img_vec[i];
		cv::Mat sample_resized;
		if (sample.size() != input_gemotry_)
			cv::resize(sample, sample_resized, input_gemotry_);
		else
			sample_resized = sample;
		cv::Mat sample_float;
		sample_resized.convertTo(sample_float, CV_32FC1);
		cv::Mat sample_normalized;
		cv::subtract(sample_float, mean_, sample_normalized);
		sample_normalized.copyTo(input_imgs->at(i));
	}
}
Classifier::~Classifier() {
	delete (caffe_net_ptr(net_));
}
