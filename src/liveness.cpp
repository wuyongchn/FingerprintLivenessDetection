//
// Created by wuyong on 19-1-24.
//

#include "liveness.hpp"

FptLiveness::FptLiveness(
    const std::string& pi_2_yml, const std::string& pi_4_yml,
    const std::string& model64x64_file, const std::string& trained64x64_file,
    const std::string& model96x96_file, const std::string& trained96x96_file) :
    enhancer_(pi_2_yml, pi_4_yml),
    cls64x64_(model64x64_file, trained64x64_file),
    cls96x96_(model96x96_file, trained96x96_file) {
}

FptLiveness::~FptLiveness() {}

float FptLiveness::Liveness(const cv::Mat& img) {
  mask_ = FptMask(img);
  rrect_ = FptRect(mask_);
  double angle = GetRotateAngle();
  cv::Mat M = cv::getRotationMatrix2D(rrect_.center, angle, 1.0);
  cv::warpAffine(img, img,  M, img.size());
  std::vector<cv::Mat> patches64x64, patches96x96;
  GetFptPatches(img, 0.5, patches64x64, patches96x96);
  std::vector<float> liveness_score64x64 = cls64x64_.Predict(patches64x64);
  std::vector<float> liveness_score96x96 = cls96x96_.Predict(patches96x96);
  float mean64x64 = 0, mean96x96=0;
  for (int i = 0; i < liveness_score64x64.size(); ++i)
    mean64x64 += liveness_score64x64[i];
  for (int i = 0; i < liveness_score96x96.size(); ++i)
    mean96x96 += liveness_score96x96[i];
  mean64x64 /= liveness_score64x64.size();
  mean96x96 /= liveness_score96x96.size();
  return 0.4 * mean64x64 + 0.6 * mean96x96;
}

cv::Mat FptLiveness::FptMask(const cv::Mat& img, const int blksze, const double thresh) {
  assert(img.channels() == 1);
  cv::Mat enhimg = enhancer_.EnhancedFpt(img);
  enhimg.convertTo(enhimg, CV_64F);
  cv::Mat mean, std_dev;
  cv::meanStdDev(enhimg, mean, std_dev);
  enhimg = (enhimg - mean.at<double>(0, 0)) / (std_dev.at<double>(0, 0) + 1e-5);
  auto new_rows = static_cast<int>(blksze * ceil(img.rows / (blksze + 1e-5)));
  auto new_cols = static_cast<int>(blksze * ceil(img.cols / (blksze + 1e-5)));
  cv::Mat padded_img = cv::Mat::zeros(new_rows, new_cols, CV_64F);
  enhimg.copyTo(padded_img(cv::Rect(0, 0, img.cols, img.rows)));
  cv::Mat stddev_img = cv::Mat::zeros(new_rows, new_cols, CV_64F);
  for (int row = 0; row < new_rows; row += blksze) {
    for (int col = 0; col < new_cols; col += blksze) {
      cv::meanStdDev(padded_img(cv::Rect(col, row, blksze, blksze)), mean, std_dev);
      for (int i = row; i < row + blksze; ++i)
        for (int j = col; j < col + blksze; ++j)
          stddev_img.at<double>(i, j) = std_dev.at<double>(0, 0);
    }
  }
  stddev_img(cv::Rect(0, 0, img.cols, img.rows)).copyTo(stddev_img);
  cv::Mat mask(img.rows, img.cols, CV_8U);
  for (int i = 0; i < mask.rows; ++i)
    for (int j = 0; j < mask.cols; ++j)
      mask.at<uchar>(i, j) = uchar(stddev_img.at<double>(i, j) > thresh);
  return mask;
}

cv::RotatedRect FptLiveness::FptRect(const cv::Mat &mask) {
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  int max_cnt_idx = 0;
  for (int i =1; i < contours.size(); ++i) {
    if (contours[i].size() > contours[max_cnt_idx].size())
      max_cnt_idx = i;
  }
  cv::RotatedRect min_ellipse = cv::fitEllipse(cv::Mat(contours[max_cnt_idx]));
  return min_ellipse;
}

double FptLiveness::GetRotateAngle() {
  rrect_.points(rect_points_);
  double l0_1 = pow(rect_points_[0].x - rect_points_[1].x, 2) + pow(rect_points_[0].y - rect_points_[1].y, 2);
  double l1_2 = pow(rect_points_[1].x - rect_points_[2].x, 2) + pow(rect_points_[1].y - rect_points_[2].y, 2);
  double delta_x, delta_y;
  if (l0_1 >= l1_2) {
	delta_x = rect_points_[0].x - rect_points_[1].x;
	delta_y = rect_points_[0].y - rect_points_[1].y;
  } else {
	delta_x = rect_points_[1].x - rect_points_[2].x;
	delta_y = rect_points_[1].y - rect_points_[2].y;
  }
  if (delta_y < 0) {
    delta_x = -delta_x;
    delta_y = -delta_y;
  }
  double angle = atan2(delta_y , delta_x) * 180 / CV_PI;
  if (angle >=0)
    angle = -(90 - angle);
  else
    angle = 90 + angle;
  return angle;
}

void FptLiveness::GetFptPatches(const cv::Mat& rotated_img, const double overlap,
                                std::vector<cv::Mat>& patches64x64, std::vector<cv::Mat>& patches96x96) {
  cv::Point2f& center = rrect_.center;
  double width = MIN(rrect_.size.width, rrect_.size.height);
  double height = MAX(rrect_.size.width, rrect_.size.height);
  double row_start = MAX(0, rrect_.center.y - height / 2);
  double col_start = MAX(0, rrect_.center.x - width / 2);
  double row_end = MIN(rotated_img.rows, rrect_.center.y + height / 2);
  double col_end = MIN(rotated_img.cols, rrect_.center.x + width / 2);
  double out_overlap = MAX(1 - overlap, 0);
  patches64x64.clear();
  for (int row = int(floor(center.y - 64 * out_overlap)); row <= int(floor(center.y + 64 * out_overlap)); row += 64 * out_overlap) {
	  for (int col = int(floor(center.x - 64 * out_overlap)); col <= int(floor(center.x + 64 * out_overlap)); col += 64 * out_overlap) {
      // Borader Check
	  cv::Rect rect;
	  rect.x = int(MIN(MAX(col_start, col - 32), col_end - 64));
	  rect.y = int(MIN(MAX(row_start, row - 32), row_end - 64));
	  rect.width = int(MIN(64, col_end - rect.x));
	  rect.height = int(MIN(64, row_end - rect.y));
      cv::Mat patch = rotated_img(rect).clone();
	  //std::cout << rect << " ";
      patches64x64.push_back(patch);
    }
  }
  patches96x96.clear();
  for (int row = int(floor(center.y - 96 * out_overlap / 2)); row <= int(floor(center.y + 96 * out_overlap / 2)); row += 96 * out_overlap) {
	  for (int col = int(floor(center.x - 96 * out_overlap / 2)); col <= int(floor(center.x + 96 * out_overlap / 2)); col += 96 * out_overlap) {
	  cv::Rect rect;
	  rect.x = int(MIN(MAX(col_start, col - 48), col_end - 96));
	  rect.y = int(MIN(MAX(row_start, row - 48), row_end - 96));
	  rect.width = int(MIN(96, col_end - rect.x));
	  rect.height = int(MIN(96, row_end - rect.y));
	  cv::Mat patch = rotated_img(rect).clone();
	  //std::cout << rect << " ";
      patches96x96.push_back(patch);
    }
  }
}