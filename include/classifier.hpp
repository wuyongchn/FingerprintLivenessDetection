#ifndef FPTLIVENESS_CLASSIFIER_HPP_
#define FPTLIVENESS_CLASSIFIER_HPP_

#include <iostream>
#include <vector>
#include <string>

#include "opencv2/opencv.hpp"

class Classifier {
public:
	Classifier(const std::string& model_file, const std::string& trained_file);
	std::vector<float> Predict(const std::vector<cv::Mat> &img_vec);
	~Classifier();
private:
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const std::vector<cv::Mat>& img_vec, std::vector<cv::Mat> *input_channels);

private:
	void* net_;
	int num_input_;
	cv::Size input_gemotry_;
	double mean_;
};
#endif  // FPTLIVENESS_CLASSIFIER_HPP_