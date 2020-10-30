#include <filesystem>
#include <iostream>

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#define IMAGE_PER_CLASS 60

namespace fs = std::filesystem;

using std::cerr;
using std::cout;
using std::endl;
using std::string;

void computeFeatures(const string &class_path, int class_label,
                     std::vector<cv::Mat> &features,
                     std::vector<int> &feature_labels) {

  cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SiftFeatureDetector::create();
  cv::Mat input_image;

  for (auto &p : fs::directory_iterator(class_path)) {
    input_image = cv::imread(p.path(), cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descritors;
    sift->detectAndCompute(input_image, cv::noArray(), keypoints, descritors);
    features.push_back(descritors);
    feature_labels.push_back(class_label);
  }
}

void computeWords(std::vector<cv::Mat> &features, int number_words,
                  cv::Mat &best_labels, cv::Mat &centers) {

  cv::TermCriteria criteria =
      cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1e3, 1e-4);
  int attempts = 10;
  cv::kmeans(features, number_words, best_labels, criteria, attempts,
             cv::KMEANS_PP_CENTERS, centers);
}

void getHistogram(std::vector<cv::Mat> &featuers, cv::Mat &centers,
                  const int number_words, std::vector<cv::Mat> &train_data) {
  cv::FlannBasedMatcher matcher;

  for (auto &desc : featuers) {
    std::vector<cv::DMatch> matches;
    matcher.match(desc, centers, matches);

    cv::Mat histo = cv::Mat::zeros(1, number_words, CV_32F);
    for (const auto &mtch : matches) {
      histo.at<float>(0, mtch.trainIdx) += 1;
    }

    train_data.push_back(histo);
  }
}

int main() {

  std::vector<cv::Mat> features;
  std::vector<int> feature_labels;
  computeFeatures("dataset/classes/chair", 1, features, feature_labels);
  computeFeatures("dataset/classes/lotus", 2, features, feature_labels);

  const int number_words = 200;
  cv::Mat best_labels, centers;
  computeWords(features, number_words, best_labels, centers);
  std::vector<cv::Mat> train_data;
  getHistogram(features, centers, number_words, train_data);

  return 0;
}