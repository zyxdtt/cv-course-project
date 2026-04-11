#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace cv;
using namespace std;

Mat getDarkChannel(const Mat& img, int winSize);
Mat estimateTransmission(const Mat& dark, double omega, double A);
Mat guideFilter(const Mat& guide, const Mat& src, int r, double eps);
Mat recoverImage(const Mat& img, const Mat& t, double A, double t0);
double computeHazeScore(const Mat& src);
double computeLowlightScore(const Mat& src);
double computeUnevenScore(const Mat& src);

//Single image dehazing algorithm based on Dark Channel Prior
//He Kaiming CVPR 2009
//omega:defogging level
//t0:lower limit of transmittance
Mat dehazeImage(const Mat& src, int winSize = 15, double omega = 0.95, double t0 = 0.1) {
	Mat img;
	src.convertTo(img, CV_64FC3, 1.0 / 255.0);
	Mat dark = getDarkChannel(img, winSize);
	int px_cnt = dark.rows * dark.cols;
	int topn = max(1, (int)(px_cnt * 0.001));//at least 1
	vector<pair<double, Point>> px;
	px.reserve(px_cnt);
	for (int i = 0; i < dark.rows; i++) {
		for (int k = 0; k < dark.cols; k++) {
			px.emplace_back(dark.at<double>(i, k), Point(k, i));
			//Point must reverse!
		}
	}
	sort(px.begin(), px.end(), [](const auto& a, const auto& b) {
		return a.first > b.first;
		});
	Vec3d A(0, 0, 0);
	for (int k = 0; k < topn; k++) {
		Point p = px[k].second;
		Vec3d val = img.at<Vec3d>(p.y, p.x);
		for (int c = 0; c < 3; c++) A[c] = max(A[c], val[c]);
	}
	//use average
	double mean = (A[0] + A[1] + A[2]) / 3.0;
	Mat t = estimateTransmission(dark, omega, mean);
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray, CV_64FC1, 1.0 / 255.0);
	Mat t_re = guideFilter(gray, t, 40, 0.001);//40 r,0.001 re
	//I=J*t+A*(1-t) => J = (I - A)/t + A
	Mat result = recoverImage(img, t_re, mean, t0);
	Mat out;
	result.convertTo(out, CV_8UC3, 255.0);
	return out;
}

Mat getDarkChannel(const Mat& img, int winSize) {
	Mat dark(img.size(), CV_64FC1);
	for (int i = 0; i < img.rows; i++) {
		for (int k = 0; k < img.cols; k++) {
			Vec3d pix = img.at<Vec3d>(i, k);
			double mi = min({ pix[0],pix[1],pix[2] });
			dark.at<double>(i, k) = mi;
		}
	}
	Mat kernel = getStructuringElement(MORPH_RECT, Size(winSize, winSize));
	morphologyEx(dark, dark, MORPH_ERODE, kernel);
	return dark;
}

Mat estimateTransmission(const Mat& dark, double omega, double A) {
	Mat t(dark.size(), CV_64FC1);
	for (int i = 0; i < dark.rows; i++) {
		for (int k = 0; k < dark.cols; k++) {
			double val = dark.at<double>(i, k);
			//t = 1 - ω * (dark / A)
			t.at<double>(i, k) = 1.0 - omega * (val / (A + 1e-6));
		}
	}
	return t;
}

Mat recoverImage(const Mat& img, const Mat& t, double A, double t0) {
	Mat j(img.size(), CV_64FC3);
	for (int i = 0; i < img.rows; i++) {
		for (int k = 0; k < img.cols; k++) {
			double tx = max(t.at<double>(i, k), t0);//restrain
			Vec3d I = img.at<Vec3d>(i, k);
			Vec3d J;
			for (int c = 0; c < 3; c++) {
				J[c] = (I[c] - A) / tx + A;
				J[c] = clamp(J[c], 0.0, 1.0);
			}
			j.at<Vec3d>(i, k) = J;
		}
	}
	return j;
}

Mat guideFilter(const Mat& guide, const Mat& src, int r, double eps) {
	Mat I = guide, p = src;
	Mat meanI, meanp;
	boxFilter(I, meanI, CV_64FC1, Size(r, r));
	boxFilter(p, meanp, CV_64FC1, Size(r, r));
	Mat meanIP, covIP;
	Mat IP = I.mul(p);
	boxFilter(IP, meanIP, CV_64FC1, Size(r, r));
	covIP = meanIP - meanI.mul(meanp);
	Mat meanII, varI;
	Mat II = I.mul(I);
	boxFilter(II, meanII, CV_64FC1, Size(r, r));
	varI = meanII - meanI.mul(meanI);
	Mat a, b;
	Mat varI_eps;
	add(varI, eps, varI_eps);
	divide(covIP, varI_eps, a);//a = cov / (var + eps)
	b = meanp - a.mul(meanI);//b = E(p) - a*E(I)
	Mat meana, meanb;
	boxFilter(a, meana, CV_64FC1, Size(r, r));
	boxFilter(b, meanb, CV_64FC1, Size(r, r));
	Mat q = meana.mul(I) + meanb;
	return q;
}

//Low-light image enhancement 
//(adaptive gamma correction, CLAHE, saturation compensation)
Mat lowlightEnhance(const Mat& src, double claheClipLimit = 2.0,
	double satCompensation = 1.2, bool denoise = true) {
	Mat hsv;
	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(hsv, channels);
	Mat V = channels[2];   
	Scalar meanV = mean(V);
	double meanBright = meanV[0] / 255.0;
	double gamma = 0.5 + (1.0 - meanBright) * 0.5;   
	gamma = clamp(gamma, 0.4, 1.2);
	Mat lut(1, 256, CV_8UC1);
	uchar* p = lut.ptr();
	for (int i = 0; i < 256; ++i) {
		double v = i / 255.0;
		v = pow(v, gamma);
		p[i] = saturate_cast<uchar>(v * 255.0);
	}
	Mat V_gamma;
	LUT(V, lut, V_gamma);
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(claheClipLimit);     
	clahe->setTilesGridSize(Size(8, 8));      
	Mat V_clahe;
	clahe->apply(V_gamma, V_clahe);
	Mat V_denoised;
	if (denoise) bilateralFilter(V_clahe, V_denoised, 9, 75, 75);
	else V_denoised = V_clahe;
	V_denoised.copyTo(channels[2]);
	Mat& S = channels[1];
	for (int i = 0; i < S.rows; ++i) {
		uchar* srow = S.ptr<uchar>(i);
		for (int j = 0; j < S.cols; ++j) {
			int newSat = saturate_cast<int>(srow[j] * satCompensation);
			srow[j] = saturate_cast<uchar>(min(newSat, 255));
		}
	}
	merge(channels, hsv);
	Mat result;
	cvtColor(hsv, result, COLOR_HSV2BGR);
	return result;
}

//Uneven Illumination Correction 
//(Based on Block Brightness Equalization and Bilinear Interpolation Smoothing Compensation)
Mat unevenIlluminationEnhance(const Mat& src, double strength = 1.5) {
	Mat lab;
	cvtColor(src, lab, COLOR_BGR2Lab);
	vector<Mat> channels;
	split(lab, channels);
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(2.0);
	clahe->setTilesGridSize(Size(8, 8));
	Mat L_enhanced;
	clahe->apply(channels[0], L_enhanced);
	L_enhanced.convertTo(L_enhanced, CV_8UC1, strength, 0);
	L_enhanced.copyTo(channels[0]);
	merge(channels, lab);
	Mat result;
	cvtColor(lab, result, COLOR_Lab2BGR);
	return result;
}

int interactive() {
	string filename;
	cout << "Enter image filename: ";
	cin >> filename;
	Mat img = imread(filename);
	if (img.empty()) {
		cout << "Cannot read image: " << filename << endl;
		return -1;
	}
	cout << "========================================" << endl;
	cout << "        Image Enhancement System" << endl;
	cout << "========================================" << endl;
	cout << "  1 - Dehaze (foggy image)" << endl;
	cout << "  2 - Low-light enhancement" << endl;
	cout << "  3 - Uneven illumination correction" << endl;
	cout << "========================================" << endl;
	cout << "Enter number (1/2/3): ";
	int type;
	cin >> type;
	Mat result;
	switch (type) {
	case 1:
		result = dehazeImage(img, 15, 0.85, 0.15);
		cout << "Processing: Dehaze" << endl;
		break;
	case 2:
		result = lowlightEnhance(img, 3.5, 1.15, false);
		cout << "Processing: Low-light enhancement" << endl;
		break;
	case 3:
		result = unevenIlluminationEnhance(img);
		cout << "Processing: Uneven illumination correction" << endl;
		break;
	default:
		cout << "Invalid choice" << endl;
		return -1;
	}
	imshow("Original", img);
	imshow("Enhanced", result);
	imwrite("result.jpg", result);
	cout << "Result saved as result.jpg" << endl;
	cout << "Press any key to exit..." << endl;
	waitKey(0);
	destroyAllWindows();
	return 0;
}

int video_enhancement() {
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cout << "Cannot open camera!" << endl;
		return -1;
	}
	cout << "========================================" << endl;
	cout << "     Real-time Video Enhancement" << endl;
	cout << "========================================" << endl;
	cout << "  1 - Dehaze (foggy image)" << endl;
	cout << "  2 - Low-light enhancement" << endl;
	cout << "  3 - Uneven illumination correction" << endl;
	cout << "========================================" << endl;
	cout << "Enter number (1/2/3): ";
	int type;
	cin >> type;
	namedWindow("Original", WINDOW_NORMAL);
	namedWindow("Enhanced", WINDOW_NORMAL);
	Mat frame, enhanced;
	cout << "Press 'q' to quit, 's' to save current frame" << endl;
	int frameCount = 0;
	double fps = 0;
	auto lastTime = chrono::steady_clock::now();
	while (true) {
		auto frameStart = chrono::steady_clock::now(); 
		cap >> frame;
		if (frame.empty()) {
			cout << "Cannot read frame!" << endl;
			break;
		}
		switch (type) {
		case 1:
			enhanced = dehazeImage(frame, 15, 0.85, 0.15);
			break;
		case 2:
			enhanced = lowlightEnhance(frame, 3.5, 1.15, false);
			break;
		case 3:
			enhanced = unevenIlluminationEnhance(frame, 1.5);
			break;
		default:
			enhanced = frame;
			break;
		}
		imshow("Original", frame);
		imshow("Enhanced", enhanced);
		frameCount++;
		auto now = chrono::steady_clock::now();
		double elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastTime).count();
		if (elapsed >= 1000.0) {
			fps = frameCount * 1000.0 / elapsed;
			cout << "[FPS] " << fixed << setprecision(1) << fps << " fps" << endl;
			frameCount = 0;
			lastTime = now;
		}
		auto frameEnd = chrono::steady_clock::now();
		double frameTime = chrono::duration_cast<chrono::milliseconds>(frameEnd - frameStart).count();
		cout << "Frame time: " << frameTime << " ms" << "    \r" << flush;
		char key = waitKey(1);
		if (key == 'q') {
			break;
		}
		else if (key == 's') {
			string filename = "capture_" + to_string(time(nullptr)) + ".jpg";
			imwrite(filename, enhanced);
			cout << "\nSaved: " << filename << endl;
		}
	}
	cap.release();
	destroyAllWindows();
	return 0;
}

Mat addTextLabel(const Mat& src, const string& text, Scalar color = Scalar(0, 255, 0)) {
	Mat result = src.clone();
	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.6;
	int thickness = 2;
	int baseline = 0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	rectangle(result, Point(5, 5), Point(textSize.width + 15, textSize.height + 15),
		Scalar(0, 0, 0), FILLED);
	putText(result, text, Point(10, textSize.height + 10),
		fontFace, fontScale, color, thickness);

	return result;
}

int multi_generation() {
	string filename;
	cout << "Enter image filename: ";
	cin >> filename;
	Mat img = imread(filename);
	if (img.empty()) {
		cout << "Cannot read image: " << filename << endl;
		return -1;
	}
	cout << "========================================" << endl;
	cout << "       Parameter Comparison Tool" << endl;
	cout << "========================================" << endl;
	cout << "  1 - Dehaze parameter comparison" << endl;
	cout << "  2 - Lowlight parameter comparison" << endl;
	cout << "  3 - Uneven illumination comparison" << endl;
	cout << "========================================" << endl;
	cout << "Enter number (1/2/3): ";
	int type;
	cin >> type;
	Mat result1, result2, result3, result4;
	Mat labeled1, labeled2, labeled3, labeled4;
	switch (type) {
	case 1: {
		cout << "\nDehaze parameter comparison:" << endl;
		result1 = dehazeImage(img, 15, 0.75, 0.15);
		result2 = dehazeImage(img, 15, 0.85, 0.15);
		result3 = dehazeImage(img, 15, 0.95, 0.15);
		result4 = dehazeImage(img, 15, 0.85, 0.20);
		labeled1 = addTextLabel(result1, "omega=0.75, t0=0.15", Scalar(0, 255, 0));
		labeled2 = addTextLabel(result2, "omega=0.85, t0=0.15", Scalar(0, 255, 0));
		labeled3 = addTextLabel(result3, "omega=0.95, t0=0.15", Scalar(0, 255, 0));
		labeled4 = addTextLabel(result4, "omega=0.85, t0=0.20", Scalar(0, 255, 0));
		Mat topRow, bottomRow, combined;
		hconcat(labeled1, labeled2, topRow);
		hconcat(labeled3, labeled4, bottomRow);
		vconcat(topRow, bottomRow, combined);
		imshow("Original", img);
		imshow("Dehaze Parameter Comparison", combined);
		imwrite("compare_dehaze.jpg", combined);
		break;
	}
	case 2: {
		cout << "\nLowlight parameter comparison:" << endl;
		result1 = lowlightEnhance(img, 2.5, 1.1, false);
		result2 = lowlightEnhance(img, 3.5, 1.15, false);
		result3 = lowlightEnhance(img, 4.5, 1.2, false);
		result4 = lowlightEnhance(img, 3.5, 1.15, true);
		labeled1 = addTextLabel(result1, "clip=2.5, sat=1.1", Scalar(0, 255, 0));
		labeled2 = addTextLabel(result2, "clip=3.5, sat=1.15", Scalar(0, 255, 0));
		labeled3 = addTextLabel(result3, "clip=4.5, sat=1.2", Scalar(0, 255, 0));
		labeled4 = addTextLabel(result4, "clip=3.5, sat=1.15, denoise", Scalar(0, 255, 0));
		Mat topRow, bottomRow, combined;
		hconcat(labeled1, labeled2, topRow);
		hconcat(labeled3, labeled4, bottomRow);
		vconcat(topRow, bottomRow, combined);
		imshow("Original", img);
		imshow("Lowlight Parameter Comparison", combined);
		imwrite("compare_lowlight.jpg", combined);
		break;
	}
	case 3: {
		cout << "\nUneven illumination parameter comparison:" << endl;

		result1 = unevenIlluminationEnhance(img, 1.2);
		result2 = unevenIlluminationEnhance(img, 1.5);
		result3 = unevenIlluminationEnhance(img, 1.8);
		result4 = unevenIlluminationEnhance(img, 2.0);

		labeled1 = addTextLabel(result1, "strength=1.2", Scalar(0, 255, 0));
		labeled2 = addTextLabel(result2, "strength=1.5", Scalar(0, 255, 0));
		labeled3 = addTextLabel(result3, "strength=1.8", Scalar(0, 255, 0));
		labeled4 = addTextLabel(result4, "strength=2.0", Scalar(0, 255, 0));

		Mat topRow, bottomRow, combined;
		hconcat(labeled1, labeled2, topRow);
		hconcat(labeled3, labeled4, bottomRow);
		vconcat(topRow, bottomRow, combined);

		imshow("Original", img);
		imshow("Uneven Parameter Comparison", combined);
		imwrite("compare_uneven.jpg", combined);
		break;
	}
	default:
		cout << "Invalid choice" << endl;
		return -1;
	}

	cout << "\nComparison image saved" << endl;
	cout << "Press any key to exit..." << endl;
	waitKey(0);
	destroyAllWindows();

	return 0;
}

int main() {
	cout << "========================================" << endl;
	cout << "   Image Enhancement System - Main Menu" << endl;
	cout << "========================================" << endl;
	cout << "  1 - Interactive single image enhancement" << endl;
	cout << "  2 - Real-time camera enhancement" << endl;
	cout << "  3 - Parameter comparison (generate 2x2 grid)" << endl;
	cout << "========================================" << endl;
	cout << "Enter your choice (1/2/3): ";

	int choice;
	cin >> choice;

	switch (choice) {
	case 1:
		return interactive();
	case 2:
		return video_enhancement();
	case 3:
		return multi_generation();
	default:
		cout << "Invalid choice. Exiting." << endl;
		return -1;
	}
}
