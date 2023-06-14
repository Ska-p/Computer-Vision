#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class ransac_est
{
public:
    // Function to compute the affine transformation given a set of three correspondences
    void computeAffineTransformation(vector<Point2f>& patch, vector<Point2f>& img, Mat& estimate);

    //  Function to compute the error between transformed points and target points
    int countConsistentCorrespondences(vector<Point2f>& patchPoints, vector<KeyPoint>& corrupted_keypoints, vector<DMatch>& good_matches, Mat& transformMatrix, double threshold);

    //  Function to estimate affine transformation using RANSAC
    void estimateAffineTransformationRANSAC(vector<DMatch> good_matches, vector<KeyPoint>& patch_keypoints, vector<KeyPoint>& corrupted_keypoints,
                                            vector<Point2f>& patch, vector<Point2f>& img, Mat& estimate, int& maxConsensusSet);

    //  Function to extract a set of 3 correspondences between the keypoints of the patch and the corrupted image
    void selectPoints(vector<DMatch>& matches, vector<KeyPoint>& patchKeypoints, vector<KeyPoint>& corruptedKeypoints, 
                      vector<cv::Point2f>& patchPoints, vector<Point2f>& corruptedPoints);
};