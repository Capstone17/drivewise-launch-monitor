
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "histogram.h"

int main() {

    // Read input image
    cv::Mat image = cv::imread("test.jpg", 0);
    if (!image.data) return 0;

    // Display the image
    cv::namedWindow("Image");
    cv::imshow("Image", image);
    
    // Histogram object
    HistogramID h;

    // Compute the histogram
    cv::Mat histo = h.getHistogram(image);

    // Loop over each bin  
    for (int i = 0; i < 256; i++) {
        std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl; 
    }

    cv::waitKey();
    return 0;

	
}
