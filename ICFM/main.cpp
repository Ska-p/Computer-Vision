#include <iostream>
#include <opencv2/opencv.hpp>
#include "ransac_est.h"

using namespace std;
using namespace cv;


void 
overlayImages(Mat& src, const Mat& patch, const Mat& homography) {
    //  By using the warpPerspective function the patch is positioned in the correct position according to the homography Matrix computed before
    //  BORDER_TRANSPARENT avoids that the pixels outside from the patch region are setted to 0 value, corresponding to black
    Mat imgTrainWarped;
    warpPerspective(patch, src, homography, src.size(), INTER_LINEAR, BORDER_TRANSPARENT);
}

//
void 
ratio_test(vector<vector<DMatch>>& matches, vector<DMatch>& goodMatches, int ratio) {
    // Filter the matches using the ratio test
    // First search for the minDistance
    double minDistance = DBL_MAX;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < minDistance) {
            minDistance = matches[i][0].distance; 
        }
    }
    // Refine by using the ratio
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio * minDistance) {
            goodMatches.push_back(matches[i][0]);
        }
    }
}

void
ratio_test_consistent(vector<vector<DMatch>>& matches, vector<DMatch>& goodMatches, int ratio) {
    // Filter the matches using the ratio test
    // First search for the minDistance
    double minDistance = DBL_MAX;
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < minDistance) {
            minDistance = matches[i][0].distance;
        }
    }
    // Refine by using the ratio
    for (int i = 0; i < matches.size(); i++) {
        if (matches[i][0].distance < ratio * minDistance && (matches[i][0].distance / matches[i][1].distance <= 0.8)) {
            goodMatches.push_back(matches[i][0]);
        }
    }
}

void 
checkMatches(vector<DMatch>& goodMatches, vector<KeyPoint>& patchKeypoints, vector<KeyPoint>& corrupted_keypoints) {
    vector<DMatch> temp;
    for (size_t i = 0; i < goodMatches.size(); i++)
    {
        DMatch tmp = goodMatches[i];
        if (tmp.queryIdx < patchKeypoints.size() && tmp.trainIdx < corrupted_keypoints.size())
        {
            temp.push_back(goodMatches[i]);
        }
    }
    goodMatches = temp;
}

string
selectPath(int folderPath) {
    if (folderPath == 1) {
        return "starwars";
    }
    else if (folderPath == 2) {
        return "pratodellavalle";
    }
    else if (folderPath == 3) {
        return "venezia";
    }
    else if (folderPath == 4) {
        return "scrovegni";
    }
    else if (folderPath == 5) {
        return "international";
    }
}

void
loadPatches(string path, vector<String> patches_name, vector<Mat>& patches) {
    glob(path, patches_name, false);
    for (size_t i = 0; i < patches_name.size(); i++)
        patches.push_back(imread(patches_name[i]));
}

int 
main(int argc, char** argv) {
    
    //  Input the folder path
    string folderPath;
    printf("Select folder a folder: \n %i. star wars \n %i. prato della valle \n %i. venezia \n %i. scrovegni \n %i. international \nChoice: ", 1, 2, 3, 4, 5);
    cin >> folderPath;
    string path = selectPath(stoi(folderPath));

    //  Ratio input
    string ratio_s;
    int ratio = 3;
    cout << "Enter a ratio value for refine step (3 by default): ";
    cin.ignore();
    getline(cin, ratio_s);
    if (!ratio_s.empty()) {
        ratio = stoi(ratio_s);
    }
    //  CORRUTPED IMAGE LOADING
    //  Read the images in the folder
    String base_image_path = "../" + path + "/image_to_complete.jpg"; 
    Mat corrupted_image = imread(base_image_path);

    //  PATCH LOADING
    //  Load patches in two different vector, to differentiate between the original patches and the modified patches
    vector<String> patches_;
    vector<Mat> patches, patches_t;

    //  GOOD PATCHES
    string patches_path = "../" + path + "/patch_?.jpg";
    loadPatches(patches_path, patches_, patches);

    //  SIFT ON CORRUPTED IMAGE
    //  Create a SIFT feature detector and descriptor extractor
    Ptr<SIFT> sift = SIFT::create();
    //  Detect and extract the SIFT features from the corrupted image
    vector<KeyPoint> corrupted_keypoints;
    Mat corrupted_descriptors;
    sift->detectAndCompute(corrupted_image, noArray(), corrupted_keypoints, corrupted_descriptors);
 
    //  GOOD PATCHES MATCHING
    for (Mat patch : patches) {

        //  Vectors for patch keypoints and descriptors
        vector<KeyPoint> patchKeypoints;
        Mat patchDescriptors;

        //  Perform SIFT on the patch
        sift->detectAndCompute(patch, noArray(), patchKeypoints, patchDescriptors);

        //  Compute the match between the image and patch features.
        //  For this, OpenCV offers the cv::BFMatcher class.
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> matches;
        matcher.knnMatch(patchDescriptors, corrupted_descriptors, matches, 2);

        //  Refine the matches found above by selecting the matches with distance less than ratio * min_distance, 
        //  where ratio is a user defined threshold and min_distance is the minimum distance found among the matches.
        vector<DMatch> goodMatches;
        ratio_test(matches, goodMatches, ratio);

        //  Draw the matches
        Mat imgMatches; 
        drawMatches(patch, patchKeypoints, corrupted_image, corrupted_keypoints, goodMatches, imgMatches);

        //  RANSAC implementation 
        //  variable initialization
        ransac_est estimator;
        vector<Point2f> ransac_points_patch;
        vector<Point2f> ransac_points_img;
        //  Compute the estimated affine transformation RANSAC by repeating 100 times the estimation
        Mat estimated_H;       
        int maxConsensousSet = 0;
        for (int i = 0; i < 100; i++)
        {
            ransac_points_patch.clear();
            ransac_points_img.clear();
            //  Compute affine trasnform estimation
            estimator.estimateAffineTransformationRANSAC(goodMatches, patchKeypoints, corrupted_keypoints, ransac_points_patch, ransac_points_img, estimated_H, maxConsensousSet);
        }
        //  Overlay the patch to the corrupted image by using the estimated H matrix
        overlayImages(corrupted_image, patch, estimated_H);
        
        //  Show matches between patch key points and corrupted keypoints
        imshow("Showing matches between unmodified patch and image", imgMatches);
        waitKey(0);
    }

    destroyWindow("Showing matches between unmodified patch and image");
    // Show final result
    imshow("Final result with good patches", corrupted_image);
    waitKey(0);
    destroyWindow("Final result with good patches");

    //  Reaload original image
    corrupted_image = imread(base_image_path);

    //  MODIFIED PATCHES LOAD
    patches_path = "../" + path + "/patch_t*.jpg";
    loadPatches(patches_path, patches_, patches_t);

    //  MODIFIED PATCHES MATCHING
    for (Mat patch : patches_t) {
        //  Vectors for patch keypoints and descriptors
        vector<KeyPoint> patchKeypoints;
        Mat patchDescriptors, flipped, tmp_patch;
        vector<vector<DMatch>> matches;
        vector<DMatch> goodMatches;

        //  Varibles for flip() function, 
        //  0 -> flip around x-axis
        //  1 -> flip around y-axis
        // -1 -> flip around both axis
        int flip_index = -1;
        //  Varible for while loop
        //  Turns false when good matches count > 20
        bool cont = false;

        //  Temporary structures used for assistance
        patch.copyTo(tmp_patch);
        patch.copyTo(flipped);

        //  This while loop runs since a good amount of matches is found
        //  After computing SIFT and BFMatcher between patch and corrutped image, the number of goodmatches is checked and
        //  if this number is lower than 20, the image is flipped
        while (!cont) {
            //  Since the number of possible flip is 3 (with index -1, 0, 1), once we performed all the flips if we haven't
            //  found a good amount of matches a resize of the patch is performed and the detection starts over
            //  with the resized patch
            if (flip_index <= 1) {
                //  Perform SIFT on the patch
                sift->detectAndCompute(flipped, noArray(), patchKeypoints, patchDescriptors);

                //  Compute the match between the image and patch features extracted in (2).
                //  For this, OpenCV offers you the cv::BFMatcher class.
                BFMatcher matcher(NORM_L2);
                matcher.knnMatch(patchDescriptors, corrupted_descriptors, matches, 2);

                //  Refine the matches found above by selecting the matches with distance less than ratio* min_distance, 
                //  where ratio is a user defined threshold and min_distance is the minimum distance found among the matches.
                vector<DMatch> good;
                ratio_test_consistent(matches, good, ratio);

                //  Check on goodMatches size
                if (good.size() > 20) {
                    cont = true;
                    copy(good.begin(), good.end(), back_inserter(goodMatches));
                }
                //  Flip the patch if not enough good matches
                else {
                    patchKeypoints.clear();
                    patchDescriptors.release();
                    flip(patch, flipped, flip_index);
                    flip_index++;
                }
            }
            //  Resize patch after trying all flips and not finding a good amount of matches
            else {
                resize(tmp_patch, patch, Size(patch.size().width * 1.5, patch.size().height * 1.5), INTER_LANCZOS4);
                flip_index = -1;
                patchDescriptors.release();
                patchKeypoints.clear();
            }
        }

        //  During implementation a problem with the queryIdx occured.
        //  Sometimes, the value retrieved by queryIdx as index was greater than the keypoints array size
        //  thus, launching an index exception in the random selection instructions.
        //  To correct this, after we found the goodmatches, a check is performed and eventually the entries
        //  that could cause the exception are removed
        checkMatches(goodMatches, patchKeypoints, corrupted_keypoints);
        
        //  RANSAC implementation
        ransac_est estimator;
        Mat estimated_H;
        int maxConsensousSet = 0;
        vector<Point2f> ransac_points_patch, ransac_points_img;
        //  Perform the estimation for an amount of times equal to 100 and keep the best one
        for (int i = 0; i < 100; i++)
        {
            ransac_points_patch.clear();
            ransac_points_img.clear();
            //  Compute affine trasnform estimation
            estimator.estimateAffineTransformationRANSAC(goodMatches, patchKeypoints, corrupted_keypoints, ransac_points_patch, ransac_points_img, estimated_H, maxConsensousSet);
            cout << "Max consistent count: " << maxConsensousSet << "\n";
        }


        /*  Used for test against RANSAC and affine implementation
        //  CODE FROM OPENCV LIBRARIES EXAMPLES - START
        //  https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
        //  Localize the object
        vector<Point2f> obj;
        vector<Point2f> scene;
        for (size_t i = 0; i < goodMatches.size(); i++)
        {
            //-- Get the keypoints from the good matches
            obj.push_back(patchKeypoints[goodMatches[i].queryIdx].pt);
            scene.push_back(corrupted_keypoints[goodMatches[i].trainIdx].pt);
        }
        Mat H = findHomography(obj, scene, RANSAC);
        //  CODE FROM OPENCV LIBRARIES EXAMPLES - END
        */


        // Draw the matches
        Mat imgMatches2;
        drawMatches(patch, patchKeypoints, corrupted_image, corrupted_keypoints, goodMatches, imgMatches2);

        // Overlay patch to the corrupted image with the estimated H matrix
        overlayImages(corrupted_image, flipped, estimated_H);

        imshow("Showing matches between modified patch and image", imgMatches2);
        waitKey(0);
    }

    destroyWindow("Showing matches between modified patch and image");
    // Show final result
    imshow("Final result with modified patches", corrupted_image);
    waitKey(0);
    destroyWindow("Final result with modified patches");
    return 0;
}