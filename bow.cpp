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
                     cv::Mat &features, std::vector<cv::Mat> &image_features,
                     std::vector<int> &feature_labels) {

  cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SiftFeatureDetector::create();
  cv::Mat input_image;

  for (auto &p : fs::directory_iterator(class_path)) {
    if (fs::is_directory(p))
      continue;
    input_image = cv::imread(p.path(), cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descritors;
    sift->detectAndCompute(input_image, cv::noArray(), keypoints, descritors);
    features.push_back(descritors);
    image_features.push_back(descritors);
    feature_labels.push_back(class_label);
  }
}

void computeWords(cv::Mat &features, const int number_words, cv::Mat &centers) {

  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1e3, 1e-4);
  int attempts = 5;
  cv::Mat best_labels;
  cv::kmeans(features, number_words, best_labels, criteria, attempts,
             cv::KMEANS_PP_CENTERS, centers);
}

void getSingleHistogramTest(cv::Mat &features, cv::Mat &centers,
                            const int number_words, cv::Mat &histo) {

  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;

  matcher.match(features, centers, matches);

  histo = cv::Mat::zeros(1, number_words, CV_32F);
  int total_documents_words = 0;
  for (const auto &mtch : matches) {
    histo.at<float>(0, mtch.trainIdx) += 1;
    total_documents_words++;
  }

  for (const auto &mtch : matches) {
    histo.at<float>(0, mtch.trainIdx) /= float(total_documents_words);
  }
}

void getSingleHistogramTrain(cv::Mat &features, cv::Mat &centers,
                             const int number_words, cv::Mat &histo,
                             int &total_words, cv::Mat &words_count_corpus) {

  cv::FlannBasedMatcher matcher;
  std::vector<cv::DMatch> matches;

  matcher.match(features, centers, matches);

  histo = cv::Mat::zeros(1, number_words, CV_32F);
  int total_documents_words = 0;
  for (const auto &mtch : matches) {
    histo.at<float>(0, mtch.trainIdx) += 1;
    words_count_corpus.at<float>(0, mtch.trainIdx) += 1.0f;
    total_documents_words++;
  }
  total_words += total_documents_words;

  for (int i = 0; i < number_words; i++) {
    histo.at<float>(0, i) /= float(total_documents_words);
  }
}

void getAllHistograms(std::vector<cv::Mat> &featuers, cv::Mat &centers,
                      const int number_words, std::vector<cv::Mat> &train_data,
                      cv::Mat &words_count_corpus) {

  int total_words_corpus = 0;

  for (auto &desc : featuers) {
    cv::Mat histo;
    getSingleHistogramTrain(desc, centers, number_words, histo,
                            total_words_corpus, words_count_corpus);
    train_data.push_back(histo);
  }

  // tf-idf: words frequency in the corpus
  for (int i = 0; i < number_words; i++) {
    words_count_corpus.at<float>(0, i) /= total_words_corpus;
    words_count_corpus.at<float>(0, i) =
        std::log(words_count_corpus.at<float>(0, i));
  }

  for (auto &histo : train_data) {
    histo = histo.dot(words_count_corpus);
  }
}

float cosine_similarity(cv::Mat &A, cv::Mat &B) {

  float dot = A.dot(B);

  return dot / (cv::norm(A) * cv::norm(B));
}

int test(std::vector<cv::Mat> &train_data, cv::Mat &test_data, cv::Mat &centers,
         const int number_words, cv::Mat &words_count_corpus,
         const std::vector<int> &feature_labels) {

  cv::Mat histo;
  getSingleHistogramTest(test_data, centers, number_words, histo);

  histo = histo.dot(words_count_corpus);

  float score = 2.0f;
  int label = -1;
  for (int i = 0; i < train_data.size(); i++) {
    float scr = cosine_similarity(test_data, train_data[i]);
    if (scr < score) {
      score = scr;
      label = feature_labels[i];
    }
  }

  return label;
}

int main(const int argc, const char **argv) {

  if (argc < 2) {
    cout << "test index is missing" << endl;
    return -1;
  }

  cv::Mat features;
  std::vector<cv::Mat> image_features;
  std::vector<int> feature_labels;

  int cls = 1;
  for (auto &p : fs::directory_iterator(argv[1])) {
    if (!fs::is_directory(p.path()))
      continue;
    cout << "computing features for class: " << p.path() << endl;
    computeFeatures(p.path(), cls++, features, image_features, feature_labels);
  }

  const int number_words = 1000;
  cv::Mat centers;

  computeWords(features, number_words, centers);
  std::vector<cv::Mat> train_data;
  cv::Mat words_count_corpus = cv::Mat::zeros(1, number_words, CV_32F);
  getAllHistograms(image_features, centers, number_words, train_data,
                   words_count_corpus);

  float acc = 0.0f;
  for (int i = 0; i < train_data.size(); i++) {
    int lbl = test(train_data, train_data[i], centers, number_words,
                   words_count_corpus, feature_labels);
    if (lbl == feature_labels[i])
      acc++;
  }

  cout << "Train accuracy: " << acc / float(train_data.size()) << endl;

  return 0;
}