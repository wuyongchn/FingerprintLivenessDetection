#include "fft_enhance.hpp"
#include "liveness.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <opencv2/core/core.hpp>


int main(int argc, char** argv) {
	std::string root = "./assets";
	FptLiveness fpt(
		root + "angular_filters_pi_2.yml", root + "angular_filters_pi_4.yml",
		root + "shufflenet_v2_1x_64x64-deploy.prototxt", root + "shufflenet_v2_1x_64x64_iter_20000.caffemodel",
		root + "shufflenet_v2_1x_96x96-deploy.prototxt", root + "shufflenet_v2_1x_96x96_iter_14000.caffemodel");

	std::string img_file = argv[1];
	cv::Mat img = cv::imread(img_file, 0);
	if (img.empty()) {
		std::cout << img_file << "img empty" << std::endl;
		return;
	}
	float score = fpt.Liveness(img);
	std::cout << img_file << " " << score << std::endl;
	return 0;
}
