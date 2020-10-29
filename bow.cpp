#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using std::cerr;
using std::cout;
using std::endl;

int main() {

  cv::Mat image, gray;
  image = cv::imread("../dataset/classes/sunflower/image_0001.jpg", 1);
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SiftFeatureDetector::create();
  std::vector<cv::KeyPoint> keypoints;
  sift->detect(gray, keypoints);

  cv::Mat output;
  cv::drawKeypoints(gray, keypoints, output);
  cv::imshow("sift keypoints", output);
  cv::waitKey(0);

  return 0;
}