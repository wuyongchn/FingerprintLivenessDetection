//
// Created by wuyong on 19-1-24.
//
#include "fft_enhance.hpp"

FFTEnhancer::FFTEnhancer(const std::string& pi_2_yml, const std::string& pi_4_yml) {
  cv::FileStorage angf(pi_2_yml, cv::FileStorage::READ);
  if (!angf.isOpened()) {
    std::cout << "can not read" << pi_2_yml << std::endl;
  }
  angf["angf"] >> angf_pi_2_;
  angf.release();
  angf.open(pi_4_yml, cv::FileStorage::READ);
  if (!angf.isOpened()) {
    std::cout << "can not read" << pi_4_yml << std::endl;
  }
  angf["angf"] >> angf_pi_4_;
  angf.release();
}

cv::Mat FFTEnhancer::EnhancedFpt(const cv::Mat &img_src) {
  assert(img_src.channels() == 1);
  cv::Mat img;
  img_src.convertTo(img, CV_64F);
  auto block_height = int(floor(img.rows - 2 * kOverlap_) / kBlockSize_);
  auto block_width = int(floor(img.cols - 2 * kOverlap_) / kBlockSize_);
  cv::Mat fft_src = cv::Mat::zeros(block_height*block_width, kFFTSize_*kFFTSize_, CV_64FC2);
  int window_size = kBlockSize_ + 2 * kOverlap_;

  // outputs
  cv::Mat oimg = cv::Mat::zeros(block_height, block_width, CV_64F);
  cv::Mat eimg = cv::Mat::zeros(block_height, block_width, CV_64F);
  cv::Mat enhimg = cv::Mat::zeros(img_src.size(), CV_64F);

  // precomputations
  cv::Mat d_mult(window_size, window_size, CV_64F);
  for (int i = 0; i < window_size; ++i)
    for (int j = 0; j < window_size; ++j)
      d_mult.at<double>(i, j) = pow(-1, (i+j));
  cv::Mat r(kFFTSize_, kFFTSize_, CV_64F);
  for (int i = 0; i < kFFTSize_; ++i)
    for (int j = 0; j < kFFTSize_; ++j)
      r.at<double>(i, j) = sqrt(pow(-kFFTSize_/2+i, 2) + pow(-kFFTSize_/2+j, 2)) + eps_;
  cv::Mat th(kFFTSize_, kFFTSize_, CV_64F);
  for (int i = 0; i < kFFTSize_; ++i)
    for (int j = 0; j < kFFTSize_; ++j) {
      double tmp = atan2(-kFFTSize_/2+i, -kFFTSize_/2+j);  // atan2(y, x)
      if (tmp < 0)
        tmp += CV_PI;
      th.at<double>(i, j) = tmp;
    }
  cv::Mat w = RaisedCosineWindow(kBlockSize_, kOverlap_);
  // Bandpass Filter
  double frq_low = kFFTSize_ / double(kRidgeMax_);
  double frq_high = kFFTSize_ / double(kRidgeMin_);
  cv::Mat dr_low(r.rows, r.cols, CV_64F), dr_high(r.rows, r.cols, CV_64F);
  for (int i = 0; i < r.rows; ++i) {
    for (int j = 0; j < r.cols; ++j) {
      double r_value = r.at<double>(i, j);
      dr_low.at<double>(i, j) = 1.0 / (1 + pow(r_value / frq_high, 4));
      dr_high.at<double>(i, j) = 1.0 / (1 + pow(frq_low / r_value, 4));
    }
  }
  cv::Mat db_pass = dr_low.mul(dr_high);
  std::vector<cv::Mat> planes;
  planes.clear();
  planes.push_back(db_pass.clone()); planes.push_back(db_pass.clone());
  cv::merge(planes, db_pass);

  // FFT Analysis
  for (int i = 0; i < block_height; ++i) {
    int row = i * kBlockSize_ + kOverlap_ + 1;
    for (int j = 0; j < block_width; ++j) {
      int col = j * kBlockSize_ + kOverlap_ + 1;
      // Extracting Local Block
      cv::Mat block = img(cv::Rect(
          col - kOverlap_ - 1, row - kOverlap_ - 1, window_size, window_size)).clone(); // Rect(x, y, w, h)
      cv::Scalar mean = cv::mean(block);
      block -= mean[0];
      block = block.mul(w);
      // Do Pre-filtering
      cv::Mat block_fft = MatlabFFT2(block.mul(d_mult), cv::Size(kFFTSize_, kFFTSize_));
      block_fft = block_fft.mul(db_pass);
      planes.clear();
      cv::split(block_fft, planes);
      cv::Mat d_energy;
      cv::magnitude(planes[0], planes[1], d_energy);
      planes.clear();
      planes.push_back(d_energy.clone()); planes.push_back(d_energy.clone());
      cv::merge(planes, d_energy);
      block_fft = block_fft.mul(d_energy);
      for (int k = 0; k < kFFTSize_*kFFTSize_; ++k)
        fft_src.at<cv::Vec2d>(block_width*i+j, k) = block_fft.at<cv::Vec2d>(k%kFFTSize_, k/kFFTSize_);
      planes.clear();
      cv::split(block_fft, planes);
      cv::magnitude(planes[0], planes[1], d_energy);
      d_energy = d_energy.mul(d_energy);
      // Compute Statistics
      mean = cv::mean(d_energy);
      oimg.at<double>(i, j) = ComputeMeanAngle(d_energy, th);
      eimg.at<double>(i, j) = log(mean[0] + eps_);
    }
  }
  // PreComputations
  cv::resize(d_mult, d_mult, cv::Size(kFFTSize_, kFFTSize_));
  for (int i = -kFFTSize_/2; i < kFFTSize_/2; ++i)
    for (int j = -kFFTSize_/2; j < kFFTSize_/2; ++j)
      d_mult.at<double>(i+16, j+16) = pow(-1, i+j);
  // Precoss the Result Maps
  for (int i = 0; i < 3; ++i)
    oimg = SmoothenOrientationImage(oimg);
  cv::Mat cimg = ComputeCoherence(oimg);
  cv::Mat bwimg = GetAngularBWImage(cimg);
  // FFT reconstruction
  for (int i = 0; i < block_height; ++i) {
    for (int j = 0; j < block_width; ++j) {
      int row = i * kBlockSize_ + kOverlap_;
      int col = j * kBlockSize_ + kOverlap_;
      cv::Mat blkfft(kFFTSize_, kFFTSize_, CV_64FC2);
      for (int k = 0; k< kFFTSize_*kFFTSize_; ++k)
        blkfft.at<cv::Vec2d>(k%32, k/32) = fft_src.at<cv::Vec2d>(i*block_width+j, k);
      cv::Mat af = GetAngularFilter(oimg.at<double>(i, j), bwimg.at<double>(i, j));
      planes.clear();
      planes.push_back(af.clone()); planes.push_back(af.clone());
      cv::merge(planes, af);
      blkfft = blkfft.mul(af);
      blkfft = MatlabIFFT2(blkfft);
      planes.clear();
      cv::split(blkfft, planes);
      blkfft = planes[0].mul(d_mult);

      blkfft(cv::Rect(kOverlap_, kOverlap_, kBlockSize_, kBlockSize_)).copyTo(
          enhimg(cv::Rect(col, row, kBlockSize_, kBlockSize_)));
    }
  }
  for (int i = 0; i < enhimg.rows; ++i) {
    for (int j = 0; j < enhimg.cols; ++j) {
      double value = enhimg.at<double>(i, j);
      enhimg.at<double>(i, j) = value >= 0 ? sqrt(value) : -sqrt(-value);
    }
  }
  cv::normalize(enhimg, enhimg, 0, 255, cv::NORM_MINMAX);
  enhimg.convertTo(enhimg, CV_8U);
  cv::resize(eimg, eimg, enhimg.size());
  for (int i = 0; i < enhimg.rows; ++i)
    for (int j = 0; j < enhimg.cols; ++j)
      if (eimg.at<double>(i, j) < kEnergyThresh_)
        enhimg.at<uchar>(i, j) = 128;
  return enhimg;
}


cv::Mat FFTEnhancer::RaisedCosineWindow(const int block_size, const int overlap) {
  int window_size = block_size + 2 * overlap;
  std::vector<double> y(window_size, 0);
  for (int i = 0; i < window_size; ++i) {
    double x = abs(window_size / 2 - i);
    if (x < double(block_size) / 2)
      y[i] = 1;
    else
      y[i] = 0.5 * (cos(CV_PI * (x - double(block_size) / 2) / overlap) + 1);
  }
  cv::Mat vec(y);
  cv::Mat w = vec * vec.t();
  return w;
}

cv::Mat FFTEnhancer::MatlabFFT2(const cv::Mat &src, const cv::Size &size) {
  int nrows = cv::getOptimalDFTSize(size.height);
  int ncols = cv::getOptimalDFTSize(size.width);
  cv::Mat padded = cv::Mat::zeros(nrows, ncols, src.type());
  src.copyTo(padded(cv::Rect(0, 0, src.cols, src.rows)));
  if (padded.channels() == 1) {
    cv::Mat planes[] = {cv::Mat_<double>(padded), cv::Mat::zeros(padded.size(), CV_64F)};
    cv::merge(planes, 2, padded);
  }
  cv::dft(padded, padded);
  return padded;
}

cv::Mat FFTEnhancer::MatlabIFFT2(const cv::Mat& src) {
  cv::Mat dst;
  src.copyTo(dst);
  cv::idft(dst, dst, cv::DFT_INVERSE+cv::DFT_SCALE);
  return dst;
}

double FFTEnhancer::ComputeMeanAngle(const cv::Mat& energy, const cv::Mat& th) {
  cv::Mat sth(th.size(), CV_64F), cth(th.size(), CV_64F);
  for (int i = 0; i < th.rows; ++i) {
    for (int j = 0; j < th.cols; ++j) {
      double value = th.at<double>(i, j);
      sth.at<double>(i, j) = sin(2*value);
      cth.at<double>(i, j) = cos(2*value);
    }
  }
  cv::Scalar num = cv::sum(energy.mul(sth));
  cv::Scalar den = cv::sum(energy.mul(cth));
  double mth = 0.5 * atan2(num[0], den[0]);
  if (mth < 0)
    mth += CV_PI;
  return mth;
}

cv::Mat FFTEnhancer::GetGaussianKernel(const int kernel_size, const double sigma0) {
  int halfSize = (kernel_size-1)/ 2;
  cv::Mat kernel(kernel_size, kernel_size, CV_64F);
  double s2 = 2.0 * sigma0 * sigma0;
  for(int i = (-halfSize); i <= halfSize; i++) {
    int m = i + halfSize;
    for (int j = (-halfSize); j <= halfSize; j++) {
      int n = j + halfSize;
      double v = exp(-(1.0*i*i + 1.0*j*j) / s2);
      kernel.at<double>(m, n) = v;
    }
  }
  cv::Scalar all = cv::sum(kernel);
  kernel.convertTo(kernel, CV_64F, (1/all[0]));
  return kernel;
}

cv::Mat FFTEnhancer::SmoothenOrientationImage(const cv::Mat& oimg) {
  cv::Mat gx(oimg.size(), CV_64F), gy(oimg.size(), CV_64F);
  for (int i = 0; i < oimg.rows; ++i) {
    for (int j = 0; j < oimg.cols; ++j) {
      double value = oimg.at<double>(i, j);
      gy.at<double>(i, j) = sin(2*value);
      gx.at<double>(i, j) = cos(2*value);
    }
  }
  cv::Mat kernel = GetGaussianKernel(5, 0.5);
  cv::filter2D(gx, gx, gx.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
  cv::filter2D(gy, gy, gy.depth(), kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
  cv::Mat noimg(oimg.size(), CV_64F);
  for (int i = 0; i < oimg.rows; ++i) {
    for (int j = 0; j < oimg.cols; ++j) {
      double nth = atan2(gy.at<double>(i, j), gx.at<double>(i, j));
      if (nth < 0)
        nth += 2 * CV_PI;
      noimg.at<double>(i, j) = 0.5 * nth;
    }
  }
  return noimg;
}

cv::Mat FFTEnhancer::ComputeCoherence(const cv::Mat& oimg) {
  cv::Mat cimg(oimg.size(), CV_64F);
  cv::Mat padded = cv::Mat::zeros(oimg.rows+4, oimg.cols+4, CV_64F);
  cv::copyMakeBorder(oimg, padded, 2, 2, 2, 2, cv::BORDER_REFLECT);
  for (int i = 3; i < padded.rows-1; ++i){
    for (int j = 3; j < padded.cols-1; ++j) {
      double th = padded.at<double>(i-1, j-1);
      cv::Mat block = padded(cv::Rect(j-3, i-3, 5, 5)).clone();
      block = abs(block - th);
      for (int k = 0; k < 25; ++k)
        block.at<double>(k/5, k%5) = cv::fast_abs(cos(block.at<double>(k/5, k%5)));
      cimg.at<double>(i-3, j-3) = cv::mean(block)[0];
    }
  }
  return cimg;
}

cv::Mat FFTEnhancer::GetAngularBWImage(const cv::Mat& cimg) {
  cv::Mat bwimg = cv::Mat::zeros(cimg.size(), CV_64F);
  bwimg += cv::Scalar::all(CV_PI / 2);
  for (int i = 0; i < bwimg.rows; ++i){
    for (int j = 0; j < bwimg.cols; ++j) {
      double value = cimg.at<double>(i, j);
      if (value <= 0.7)
        bwimg.at<double>(i, j) = CV_PI;
      else if (value >= 0.9)
        bwimg.at<double>(i, j) = CV_PI / 4;
    }
  }
  return bwimg;
}

cv::Mat FFTEnhancer::GetAngularFilter(const double to, const double bw) {
  int steps = angf_pi_2_.cols;
  double delta = CV_PI / steps;
  int i = int(floor((to + delta / 2) / delta)) % steps + 1;
  cv::Mat r = cv::Mat::zeros(kFFTSize_, kFFTSize_, CV_64F);
  if (bw == CV_PI / 4) {
    for (int k = 0; k < angf_pi_4_.rows; ++k)
      r.at<double>(k/kFFTSize_, k%kFFTSize_) = angf_pi_4_.at<double>(k, i-1);
  } else if (bw == CV_PI / 2){
    for (int k = 0; k < angf_pi_2_.rows; ++k)
      r.at<double>(k/kFFTSize_, k%kFFTSize_) = angf_pi_2_.at<double>(k, i-1);
  } else
    r += cv::Scalar::all(1);
  return r;
}