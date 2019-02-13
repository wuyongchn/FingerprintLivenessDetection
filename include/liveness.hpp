//
// Created by wuyong on 19-1-24.
//
#ifndef FPTLVENESS_LIVENESS_HPP_
#define FPTLVENESS_LIVENESS_HPP_
#define CPU_ONLY 1
#include "fft_enhance.hpp"
#include "classifier.hpp"
#include <opencv2/opencv.hpp>

class FptLiveness {
 public:
  FptLiveness(const std::string& pi_2_yml, const std::string& pi_4_yml,
              const std::string& model64x64_file, const std::string& trained64x64_file,
              const std::string& model96x96_file, const std::string& trained96x96_file);
  float Liveness(const cv::Mat& img);
	~FptLiveness();

 private:
  cv::Mat FptMask(const cv::Mat& img, const int blksze=5, const double thresh=0.085);
  cv::RotatedRect FptRect(const cv::Mat& mask);
  double GetRotateAngle();
  void GetFptPatches(const cv::Mat& rotated_img, const double overlap,
                     std::vector<cv::Mat>& patches64x64, std::vector<cv::Mat>& patches96x96);
  FFTEnhancer enhancer_;
  cv::Mat mask_;
  cv::RotatedRect rrect_;
  cv::Point2f rect_points_[4];
  Classifier cls64x64_;
  Classifier cls96x96_;
};
#endif // FPTLVENESS_LIVENESS_HPP_
