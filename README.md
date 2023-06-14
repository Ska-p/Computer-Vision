# Image Completion with Features Matching
##  Final project for Computer Vision course
A <a href=#tabs> Corrupted Image </a> is given in input together with a set of patches that includes also manipulated patches. The manipulated patches have been corrupted with noise, flipped and resized. The aim of the project is to correctly overlay the patches over the corrupted zones of the images. Results and short explanation in [Report]https://github.com/Ska-p/Computer-Vision/blob/main/ICFM/Report.pdf.
To solve the problem Scale Invariant Feature Transform (SIFT) and BFMatcher classes have been used for feature extraction and matching. Also, an Affine Transformation matrix and RANdom SAmple Consenseous (RANSAC) algorithm implementation is used to estimate the homography matrix. This allows to overlay the patches in the correct location.
Some problems with the manipulated patches occurs in the more complex images.

<a name="tabs">![Tabs](https://github.com/Ska-p/Computer-Vision/assets/102731992/a855c228-8aea-48e8-b59e-c64be721171c)</a>
