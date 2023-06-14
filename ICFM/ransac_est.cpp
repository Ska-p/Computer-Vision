// my_class.cpp
#include "ransac_est.h" // header in local directory
#include <iostream> // header in standard library
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void  
ransac_est::computeAffineTransformation(vector<Point2f>& patch, vector<Point2f>& img, Mat& estimate) {
    //  Extract from the set of correspondences the x,y and u.v coordinates
    double x1 = patch[0].x, y1 = patch[0].y, x2 = patch[1].x, y2 = patch[1].y, x3 = patch[2].x, y3 = patch[2].y;
    double u1 = img[0].x, v1 = img[0].y, u2 = img[1].x, v2 = img[1].y, u3 = img[2].x, v3 = img[2].y;

    //  Build the matrix A
    Mat A = (Mat_<double>(6, 6) <<  x1, y1, 0, 0, 1, 0,
                                     0, 0, x1, y1, 0, 1,
                                    x2, y2, 0, 0, 1, 0,
                                     0, 0, x2, y2, 0, 1,
                                    x3, y3, 0, 0, 1, 0,
                                     0, 0, x3, y3, 0, 1);
    //  Build the matrix/vector b
    Mat b = (Mat_<double>(6, 1) << u1, v1, u2, v2, u3, v3);

    //  Compute the inverse of A with the opencv libraries
    Mat inverse = A.inv();

    //  Compute the estimate of the Transformation matrix, x = b*A^-1
    estimate = inverse * b;
}

//  Function to compute the error between transformed points and target points
int
ransac_est::countConsistentCorrespondences(vector<Point2f>& patch_points, vector<KeyPoint>& corrupted_keypoints, vector<DMatch>& good_matches,
                                            Mat& transformMatrix, double threshold) {
    int consistentCount = 0;
    int patch_counter = 0;
    double error = 0;
    for (int i = 0; i < good_matches.size(); i++)
    {
        int j = 0;
        //  Extract the coordinates of the patch and of the correspondence in the corrupted image
        while (j < 3) {
            double x = patch_points[j].x;
            double y = patch_points[j].y;
            double u = corrupted_keypoints[good_matches[i].trainIdx].pt.x;
            double v = corrupted_keypoints[good_matches[i].trainIdx].pt.y;

            //  Apply the estimated transform matrix to the patch point
            double estimatedU = transformMatrix.at<double>(0, 0) * x + transformMatrix.at<double>(0, 1) * y + transformMatrix.at<double>(0, 2);
            double estimatedV = transformMatrix.at<double>(1, 0) * x + transformMatrix.at<double>(1, 1) * y + transformMatrix.at<double>(1, 2);

            //  Compute the error
            error = abs(estimatedU - u) + abs(estimatedV - v);
            //  If error lower than threshold increase counter
            if (error < threshold)
                consistentCount++;
            j++;
        }
    }
     return consistentCount;
}

//  Function to estimate affine transformation using RANSAC
void 
ransac_est::estimateAffineTransformationRANSAC( vector<DMatch> good_matches, vector<KeyPoint>& patch_keypoints, vector<KeyPoint>& corrupted_keypoints, 
                                                vector<Point2f>& patch_points, vector<Point2f>& corrupted_points, Mat& estimate, int& maxConsensusSet) {

    selectPoints(good_matches, patch_keypoints, corrupted_keypoints, patch_points, corrupted_points);
    //  For the algorithm to work we need exactly 3 points, thus we make a check on the size of the vector given in input
    if (patch_points.size() == 3 && corrupted_points.size() == 3) {
        double errorThreshold = 3.0;    //  Threshold for good estimation

        //  Compute affine transformation from the sampled correspondences
        Mat tmp;
        computeAffineTransformation(patch_points, corrupted_points, tmp);

        //  Build Transformation matrix
        Mat H_prime = (Mat_<double>(3, 3) <<tmp.at<double>(Point(0, 0)), tmp.at<double>(Point(0, 1)), tmp.at<double>(Point(0, 4)),
                                            tmp.at<double>(Point(0, 2)), tmp.at<double>(Point(0, 3)), tmp.at<double>(Point(0, 5)),
                                            0, 0, 1);

        int consensusSet = 0;
        //  Compute parameters to later check if this computedAffineTransformation is the best     
        //  The threshold is |delta_u| + |delta_v| < T
        consensusSet = countConsistentCorrespondences(patch_points, corrupted_keypoints, good_matches, H_prime, errorThreshold);

        //  Check if the current model is the best one found so far
        //  If it is, copy the transformation matrix into estimate to retrieve it at the end of the procedure
        if (consensusSet > maxConsensusSet) {
            maxConsensusSet = consensusSet;
            H_prime.copyTo(estimate);
        }
    }
}

void 
ransac_est::selectPoints(vector<cv::DMatch>& matches, vector<cv::KeyPoint>& patchKeypoints,
                         vector<cv::KeyPoint>& corruptedKeypoints, vector<cv::Point2f>& patchPoints,
                         vector<cv::Point2f>& corruptedPoints) {
    // Select three random matches
    set<int> indices;
    while (indices.size() < 3)
    {
        int index = rand() % matches.size();
        if(find(indices.begin(), indices.end(), index) == indices.end())
            indices.insert(index);
    }

    // Get the corresponding keypoints and corrupted points
    for (int index : indices)
    {
        DMatch match = matches[index];
        Point2f patchKeypoint = patchKeypoints[match.queryIdx].pt;
        Point2f corruptedKeypoint = corruptedKeypoints[match.trainIdx].pt;
        patchPoints.push_back(patchKeypoint);
        corruptedPoints.push_back(corruptedKeypoint);

    }
}