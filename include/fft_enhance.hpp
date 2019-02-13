//
// Created by wuyong on 19-1-24.
//

#ifndef FPTLIVENESS_FFT_ENHANCE_HPP_
#define FPTLIVENESS_FFT_ENHANCE_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

class FFTEnhancer {
 public:
//  FFTEnhancer(const std::string pi_2_yml, const std::string pi_4_yml);
  FFTEnhancer(const std::string& pi_2_yml, const std::string& pi_4_yml);
  cv::Mat EnhancedFpt(const cv::Mat &src);
 private:
  cv::Mat RaisedCosineWindow(const int block_size, const int overlap);
  cv::Mat MatlabFFT2(const cv::Mat& src, const cv::Size& size);
  cv::Mat MatlabIFFT2(const cv::Mat& src);
  double ComputeMeanAngle(const cv::Mat& energy, const cv::Mat& th);
  cv::Mat GetGaussianKernel(const int kernel_size, const double sigma0);
  cv::Mat SmoothenOrientationImage(const cv::Mat& oimg);
  cv::Mat ComputeCoherence(const cv::Mat& oimg);
  cv::Mat GetAngularBWImage(const cv::Mat& cimg);
  cv::Mat GetAngularFilter(const double to, const double bw);
  const int
      kFFTSize_      = 32,
      kBlockSize_    = 12,
      kOverlap_      = 6,
      kRidgeMin_     = 3,
      kRidgeMax_     = 18,
      kEnergyThresh_ = 6;
  const double eps_  = 1e-7;
  cv::Mat angf_pi_4_;
  cv::Mat angf_pi_2_;
};


#endif //FPTLIVENESS_FFT_ENHANCE_HPP_
