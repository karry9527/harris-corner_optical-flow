/*
建置環境： visual studio 2013, opencv 3.0

最終結果圖為u v放大100倍後的效果
*/

#include <iostream>  
#include <cmath>
#include <iomanip>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

//Gaussian kernel
void Gaussian(float kernel[][3])
{
	
	float sum = 0.0, temp, sigma = 1;
	for (int x = -1; x <= 1; x++)
		for (int y = -1; y <= 1; y++)
		{
			temp = exp(-(x*x + y*y) / (2 * pow(sigma, 2)));
			kernel[x + 1][y + 1] = temp / ((float)CV_PI * 2 * pow(sigma, 2));
			sum += kernel[x + 1][y + 1];
		}	
}


//compute Gaussian filter
void Gaussian_filter(Mat &image)
{
	
	float kernel[3][3];
	Gaussian(kernel);
	for (int j = 1; j < image.rows - 1; j++)
		for (int i = 1; i < image.cols - 1; i++)
		{
			float sum = 0;
			for (int y = 0; y < 3; y++)
				for (int x = 0; x < 3; x++)
				{

					sum += image.at<float>(j - 1 + y, i - 1 + x) * kernel[x][y];
				}
			image.at<float>(j, i) = sum;
		}

}
Mat matrix_mul(Mat A, Mat B)
{
	//Mat b(9, 1, CV_32FC1), A(9, 2, CV_32FC1), ATA, AT, ATb;
	Mat mul(A.rows, B.cols, CV_32FC1);
	float sum = 0;
	for (int j = 0; j < A.rows; j++)
		for (int k = 0; k < B.cols; k++)
		{
			for (int i = 0; i < A.cols; i++)
			{
				sum += A.at<float>(j, i) * B.at<float>(i, k);
			}
			mul.at<float>(j, k) = sum;
			sum = 0;
		}
	return mul;
}

int main()
{
	Mat img = imread("1.jpg"), next_img = imread("2.jpg"), OF_img = img.clone(), gray_img, next_gray_img;
	cvtColor(img, gray_img, CV_RGB2GRAY);
	cvtColor(next_img, next_gray_img, CV_RGB2GRAY);

	
	Mat img_grad_x, img_grad_y, next_img_grad_x, next_img_grad_y;
	float k = 0.04f, thresh;
	int scale = 1, delta = 0, ddepth = CV_32F;
	cout << "Suggest:Do not enter number too small" << endl;
	do{
		cout << "Please enter the thrshold(0~255)：";
		cin >> thresh;
	} while (!(thresh >= 0 && thresh <= 255));


	//get Ix, Iy
	Sobel(gray_img, img_grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(gray_img, img_grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	//get next_img's Ix, Iy
	Sobel(next_gray_img, next_img_grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	Sobel(next_gray_img, next_img_grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);


	Mat img_grad_xx, img_grad_yy, img_grad_xy, det, trace, response, temp;
	
	multiply(img_grad_x, img_grad_x, img_grad_xx);
	multiply(img_grad_y, img_grad_y, img_grad_yy);
	multiply(img_grad_x, img_grad_y, img_grad_xy);
	
	//Gaussian filter of Ixx, Iyy, Ixy
	
	Gaussian_filter(img_grad_xx);
	Gaussian_filter(img_grad_yy);
	Gaussian_filter(img_grad_xy);

	//compute response
	multiply(img_grad_xx, img_grad_yy, det);	//mul each elements
	multiply(img_grad_xy, img_grad_xy, temp);
	det -= temp;
	trace = img_grad_xx + img_grad_yy;
	multiply(trace, trace, trace, k);
	response = det - trace ;
	
	normalize(response, response, 0, 255, NORM_MINMAX, CV_32F, Mat());

	next_gray_img.convertTo(next_gray_img, CV_32F);
	gray_img.convertTo(gray_img, CV_32F);

	int windows_size = 9;
	cout << "win:" << windows_size / 2 << endl;
	//存取矩陣元素、存取圖片像素，行列次序剛好顛倒
	for (int j = windows_size / 2; j < response.rows - windows_size / 2; j++)
	{
		for (int i = windows_size / 2; i < response.cols - windows_size / 2; i++)
		{
			if ((int)response.at<float>(j, i) > thresh)
			{
				//circle the feature
				circle(img, Point(i, j), 5, Scalar(0, 0, 255), 2, 8, 0);

				//optical flow

				//Mat(col_size, row_size)
				//Mat.at<...>(col_index, row_index)
				Mat b(windows_size*windows_size, 1, CV_32FC1), A(windows_size*windows_size, 2, CV_32FC1), ATA, AT, ATb, uv;
				float sum_Ixx;
				//compute A and b matrix
				for (int y = 0; y < windows_size; y++)
					for (int x = 0; x < windows_size; x++)
					{
						b.at<float>(y * windows_size + x, 0) = -(next_gray_img.at<float>(j - windows_size / 2 + y, i - windows_size / 2 + x) - gray_img.at<float>(j - windows_size / 2 + y, i - windows_size / 2 + x));
						A.at<float>(y * windows_size + x, 0) = next_img_grad_x.at<float>(j - windows_size / 2 + y, i - windows_size / 2 + x);
						A.at<float>(y * windows_size + x, 1) = next_img_grad_y.at<float>(j - windows_size / 2 + y, i - windows_size / 2 + x);
						
					}
				//compute (A^T)b matrix
				AT = A.t();
				ATb = matrix_mul(AT, b);
				mulTransposed(A, ATA, true);
				//compute u and v
				uv = matrix_mul(ATA.inv(), ATb);
				float u = uv.at<float>(0, 0), v = uv.at<float>(1, 0);
				cout << "i:" << i << "  j:" << j << endl;
				cout << "u:" << u << "  v:" << v << endl;
				//scaling u and v with 100
				line(OF_img, Point(i, j), Point(i + u*100, j + v*100), Scalar(0, 255, 255), 2);
			}
		}
	}
	


	//output the pic
	imshow("origin image", img);
	imshow("next image", next_img);
	imshow("optical flow", OF_img);

	//pause
	waitKey(60000);
}

