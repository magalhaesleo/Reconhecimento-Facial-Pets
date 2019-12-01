// ReconhecimentoFacialPets.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

// CPP program to detects face in a video

// Include required header files from OpenCV directory
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include "Servo.h"


using namespace std;
using namespace cv;


void detectAndDraw(Mat& img, CascadeClassifier& cascade, CascadeClassifier& nestedCascade, double scale);
vector<int> labels;
//Ptr<face::FaceRecognizer> recognizer = face::BasicFaceRecognizer::create(15, 4000);
//Ptr<face::FaceRecognizer> recognizer = face::LBPHFaceRecognizer::create(15, 4000);
//Ptr<face::FaceRecognizer> recognizer = face::FisherFaceRecognizer::create();

cv::Ptr<cv::face::FaceRecognizer> recognizer = cv::face::createFisherFaceRecognizer();
Servo servo;

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
		printf("file: %s\n", dirp->d_name);
		 if (strcasestr(dirp->d_name, "jpg") != NULL)
		 {
			 files.push_back(string(dir).append("/").append(dirp->d_name));
		 }
    }
    closedir(dp);
    return 0;
}

static void read_images(vector<Mat>& images, vector<int>& labels)
{
	vector<string> files;
	getdir("../../images", files);

	for (string file : files)
	{
		printf("filelist: %s\n", file.c_str());
		images.push_back(imread(file, IMREAD_GRAYSCALE));
		int label = file.find("2019") != string::npos ? 2 : 1;
        labels.push_back(label);
        printf("label: %i\n", label);
	}
}


int main(int argc, const char** argv)
{

	// VideoCapture class for playing video for which faces to be detected
	VideoCapture capture;

	Mat frame, image;


	// PreDefined trained XML classifiers with facial features
	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	// Load classifiers from "opencv/data/haarcascades" directory
	nestedCascade.load("../../haarcascades/haarcascade_eye_tree_eyeglasses.xml");

	// Change path before execution
	//cascade.load("../../haarcascades/haarcascade_frontalcatface.xml");
	cascade.load("../../haarcascades/haarcascade_frontalcatface_extended.xml");

	vector<Mat> images;

	read_images(images, labels);

	for (Mat &img : images)
	{
		//img.reshape()
		cv::resize(img, img, Size(30, 30));
	}

	recognizer->train(images, labels);

	//modelEigen->save("modelSaved.xml");

	capture.open(0);

	if (capture.isOpened())
	{
		// Capture frames from video and detect faces
		cout << "Face Detection Started...." << endl;

		while (1)
		{
			capture >> frame;
			if (frame.empty())
				break;
			Mat frame1 = frame.clone();
			transpose(frame1, frame1);
			flip(frame1, frame1, 0);
			detectAndDraw(frame1, cascade, nestedCascade, scale);
			char c = (char)waitKey(1);

			// Press q to exit from window
			if (c == 27 || c == 'q' || c == 'Q')
			{
				break;
			}

		}
	}
	else
		cout << "Could not Open Camera";
	return 0;
}


void detectAndDraw(Mat& img, CascadeClassifier& cascade,
	CascadeClassifier& nestedCascade,
	double scale)
{
	vector<Rect> faces;
	Mat gray, smallImg;

	cvtColor(img, gray, COLOR_BGR2GRAY); // Convert to Gray Scale
	double fx = 1 / scale;

	// Resize the Grayscale Image
	resize(gray, smallImg, Size(), fx, fx, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	// Detect faces of different sizes using cascade classifier
	cascade.detectMultiScale(smallImg, faces, 1.1,
		3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	// Draw circles around the faces
	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect r = faces[i];
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;

		int radius;

        Mat fcR;
		cv::resize(smallImg(r), fcR, Size(30, 30));
		int predicted_label = -1;
		predicted_label = recognizer->predict(fcR);

		printf("predict: %i\n", predicted_label);

		string cat;
		Scalar color; // Color for Drawing tool
		switch (predicted_label)
		{
		case 1:
			cat = "REGISTRADO";
			color = Scalar(32,247,24);
			servo.RotateDispenser();
			break;
		default:
			cat = "NAO REGISTRADO";
			color = Scalar(0, 0, 255);
			break;

        }



		double aspect_ratio = (double)r.width / r.height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r.x + r.width*0.5)*scale);
			center.y = cvRound((r.y + r.height*0.5)*scale);
			radius = cvRound((r.width + r.height)*0.25*scale);
			circle(img, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
				Point(cvRound((r.x + r.width - 1)*scale),
					cvRound((r.y + r.height - 1)*scale)), color, 3, 8, 0);
		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(r);

		// Detection of eyes int the input image
		nestedCascade.detectMultiScale(smallImgROI, nestedObjects, 1.1, 3,
			0 | CASCADE_SCALE_IMAGE, Size(30, 30));


        cv::putText(img, cat, Point(faces[i].x, faces[i].y), FONT_HERSHEY_DUPLEX, 2, color);

		if (!nestedObjects.empty())
		{
			// Draw circles around eyes
			for (size_t j = 0; j < nestedObjects.size(); j++)
			{

				Rect nr = nestedObjects[j];
				center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
				center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
				radius = cvRound((nr.width + nr.height)*0.25*scale);
				circle(img, center, radius, color, 3, 8, 0);
			}
		}
	}

	// Show Processed Image with detected faces
	imshow("Face Detection", img);
}
