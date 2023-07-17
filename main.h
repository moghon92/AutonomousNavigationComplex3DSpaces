#include <SDL.h>
#include <Ole2.h>
#include <Windows.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>

#include "cv.h"
#include "highgui.h"
#include "cvaux.h"
#include "objdetect\objdetect.hpp"
#include "highgui\highgui.hpp"
#include "imgproc\imgproc.hpp"
#include "opencv.hpp"
#include "opencv_modules.hpp"
#include <BlobResult.h>

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctime>
#include <ctype.h>

#include <SDL_opengl.h>
#include <glut.h>
#include <freeglut.h>
#include <freeglut_ext.h>
#include <freeglut_std.h>





//.................................Constants..................................................\\ 

#define width1 640
#define height1 480
#define widthdepth 640 //320
#define heightdepth 480 //240
#define copterRealLength 800
#define copterRealHeight 250


//.................................Headers....................................................\\ 

void fillBuffer(const BYTE* c);
void fillbytebufferdepth(const USHORT* c);
void drawKinectDatadepth();
bool CannyThreshold();
bool Canny4Motion();
//bool faceDetection(Mat frame);
bool makeContours();
bool motiondetection();
bool trackColorObject();
bool BlobDetection(IplImage* orignal);
bool Canny4Depth();
bool makeContoursDepth() ;
void findSafeArea(double nearestDist);
bool drawTunnel3D(cv::Point tl,  cv::Point br);
void bubbleSort(double* a, int N);
int findElementIndexInArray(double* a, int N, double x);
double getRealDistX(int x, double Dist);
double getRealDistY(int y, double Dist);
int getYfromRealY(double y, double Dist);
int getXfromRealX(double x, double Dist);
//bool isElementInVector(vector<double> v, double element);
void extendPath();
double CubicInterpolate(double y0, double y1, double y2, double y3, double mu);
void extendPath3D();


//..............................Kinect variables...............................................\\

HANDLE rgbStream;
HANDLE rgbEvent;
HANDLE depthEvent;
HANDLE depthStream;
INuiSensor* sensor;
int angle = 0;
bool depth = false;


//.............................openCV variables..................................................\\

using namespace cv;

IplImage* motion;
IplImage* imgTracking = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
IplImage* RGBout = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
IplImage* imageRGB = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
IplImage* imgHSV = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
IplImage* imgThresh = cvCreateImage(cvGetSize(imgHSV),IPL_DEPTH_8U, 1);
IplImage* accumelatedFrames = cvCreateImage(cvSize(widthdepth, heightdepth),IPL_DEPTH_8U, 4);
IplImage *originalThr = cvCreateImage(cvSize(width1, height1),IPL_DEPTH_8U, 1);



Mat imagegrey;
Mat Depth;
Mat edges1, edges2,edges3, detected_edges, detected_edges2, detected;
Mat img_curr, img_prev, img_diff, cumulative, imageRGBmotion, drawing, drawing2;

CvMat* Depth2 = cvCreateMat(heightdepth,widthdepth,CV_64FC1);
CvMat* map = cvCreateMat(480,640,CV_64FC3);

int lowThreshold;
int ratio = 2;
int const highThreshold= ratio*lowThreshold;
int const kernel_size = 3;

int center_cany_x = 0;
int center_cany_y = 0;
bool draw = false;
bool first = true;

int lastX = -1;
int lastY = -1;
int lowerH = 0;
int lowerS = 60;
int lowerV = 70;
int upperH = 30;
int upperS = 255;
int upperV = 255;


const char* face_cascade_name = "C:\\opencv243\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
const String body_cascade_name = "C:\\opencv243\\data\\haarcascades\\haarcascade_profileface.xml";
CascadeClassifier face_cascade;
RNG rng(12345);


//blobs
CBlob *currentBlob;
IplImage *original = cvCreateImage(cvSize(widthdepth, heightdepth),IPL_DEPTH_8U, 4); 
CBlobResult blobs;
IplImage* displayedImage = cvCreateImage(cvSize(widthdepth, heightdepth),IPL_DEPTH_8U, 4);


//.............................openGL variables..................................................\\

GLfloat xRotated;
GLfloat yRotated;
GLfloat zRotated;


//......................................flags..............................................\\

bool flagEdge = true;
bool flagRGB = true;
bool flagContours = true;
bool flagContoursDepth = true;
bool flagcolorT = true;
bool flagDepth = true;
bool flagBlobs = true;


//..................................IMAGE BUFFERS...........................................\\

char buffer[width1*height1*4];
char bufferdepth[widthdepth*heightdepth*4];
//char bufferMap[1000*1000];
double bufferdepth2[widthdepth*heightdepth];
vector<double> distArray;
vector<double> obstacle2DPosArray;
vector<int> obstaclePosArray;
int boolArray[widthdepth*heightdepth];
double nearestDst;
int numberOfObstacles;
double midObjDistAvg;
vector<double> path;
vector<int> path2D;
vector<int> ExtendedPath2D;
vector<double> ExtendedPath3D;
vector<double> newExtendedPath3D;//with espect to midle o screen
vector<double> GlobalExtendedPath3D; //with respect to global frame of refrence
vector<int> AstarPath;
vector<int> ExtendedAstar;
vector<double> AstarPath3D;
vector<double> currLocation;
vector<double> currVelocities;
vector<double> Vvector;

 int Xold = 320;
 int Yold = 240;
 const double safteyFactor = 50; //(copterRealLength/1.7);


 //KALMAN FILTER//

 	double predX;
	double predY;
	double updatedX = 320;
	double updatedY = 240;
	double predCovX;
	double predCovY;
	double updatedCovX = 10;
	double updatedCovY = 10;

	int A = 1;
	int B = 1;
	int C = 1;

	int U = 0;
	double Q = 30;
	double R = 1;

	double Kx;
	double Ky;

	int kalmanInputX;
	int kalmanInputY;



	//openGL
	   float X = 0.0f;        // Translate screen to x direction (left or right)
       float Y = 0.0f;        // Translate screen to y direction (up or down)
       float Z = 0.0f;        // Translate screen to z direction (zoom in or out)
       float rotX = 0.0f;    // Rotate screen on x axis 
       float rotY = 0.0f;    // Rotate screen on y axis
       float rotZ = 0.0f;    // Rotate screen on z axis

       float rotLx = 0.0f;   // Translate screen by using the glulookAt function 
                                     // (left or right)
       float rotLy = 0.0f;   // Translate screen by using the glulookAt function 
       float rotLz = 0.0f;   // Translate screen by using the glulookAt function 
                                     // (zoom in or out)
	   bool lines = true;       // Display x,y,z lines (coordinate lines)
       bool rotation = false;   // Rotate if F2 is pressed   
       int old_x, old_y;        // Used for mouse event
       int mousePressed;

 //================================================================================================\\
//======================================= N O D E =================================================\\
//=================================================================================================\\
 
 class node
{
	int index;
	int parentIndex;
	int x;
	int y;
	double xReal;
	double yReal;
	double zReal;
    double Gcost; 
    double Fcost;
	double Hcost;

   public:

	  node() {
		   ;
	   }


	 node(int x1, int y1) 
           {x = x1; y = y1;}
    
	
	 double getXreal() {
		return xReal;
	}
	  
	double getZreal() {
		return zReal;
	}

	double getYreal() {
		return yReal;
	}

	int getX() {
		return x;
	}

	void setIndex( int i) {
		index = i;
	}

	void setParentIndex( int i) {
		parentIndex = i;
	}
	int getParentIndex( ) {
		return parentIndex ;
	}

	int getY() {
		return y;
	}

	int getIndex() {
		return index;
	}

	double getGcost() {
		return Gcost;
	}

	double getHcost() {
		return Hcost;
	}

	double getFcost() {
		return Fcost;
	}

	void setX(int xnew) {
		x = xnew;
	}

	void setY(int ynew) {
		y = ynew;
	}

	void setGcost(double newGcost) {
		Gcost = newGcost;
	}

	void setHcost(double newHcost) {
		Hcost = newHcost;
	}

	void setFcost() {
		Fcost = Gcost + Hcost;
	}

	void setXreal() {
		xReal = getRealDistX(x, zReal); //+ safteyFactor;
	}
	
	void setYreal() {
		yReal = getRealDistY(y, zReal);// + safteyFactor;
	}

	void setZreal(double z) {
		zReal = z;// - safteyFactor;
	}
};
//================================================================================================\\
//======================================= C O D E ================================================\\
//================================================================================================\\



//.........................................Kinect Intialization........................................................\\

 bool initKinect() {
	// Get a working kinect sensor
	HRESULT hr;
	int numSensors;
	if (NuiGetSensorCount(&numSensors) < 0 || numSensors < 1) return false;
	if (NuiCreateSensorByIndex(0, &sensor) < 0) return false;
	hr =sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX 
	| NUI_INITIALIZE_FLAG_USES_COLOR);
	sensor->NuiCameraElevationSetAngle(angle);
	if(!(hr == S_OK)) exit(1);

	//Initiate RGB
	rgbEvent = CreateEvent(NULL,TRUE,FALSE, NULL);
	hr =sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR,NUI_IMAGE_RESOLUTION_640x480,0,2,rgbEvent,&rgbStream);
	if(FAILED(hr))exit(2);
	
	//Initiate depth
	depthEvent = CreateEvent(NULL,TRUE,FALSE, NULL);
	hr = sensor->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH,NUI_IMAGE_RESOLUTION_640x480,0,2,depthEvent, &depthStream);
	if(FAILED(hr))exit(3);

	return sensor;
}

//...............................Fill with kinect Colour data...................................................\\

static void fillBuffer(const BYTE* c) {

        const BYTE* dataEnd = c + (width1*height1)*4;
		for(int i = 0 ; (c < dataEnd) ; i++){

			int val = (int)*c;
			//USHORT depth = NuiDepthPixelToDepth( curr2[0]);
			//exit(depth);
			/*if() val = 0x00;
			else val = 0xFF;*/
			
			
			char x = (char)(val);
			buffer[i] = x;
			c++;
		}
    }

//...............................Fill Buffer with kinect Colour data.....................................................\\

static void fillbytebufferdepth(const USHORT* c){
   const USHORT* dataEnd = c + (widthdepth*heightdepth);

	int index = -1;
	int index2 = -1;
        while (c < dataEnd) {
			index++;
			index2++;
            // Get depth in millimeters
            USHORT depth = NuiDepthPixelToDepth(*c++);
			USHORT depth2 = depth;
			
			//depth = (depth - 800) * (255 - 0) / (4000 - 800) + 0;

			if( depth > 3000) {
				 depth = 0x00;
				 depth2 = 3000;

			}
			else if(depth < 900) {
				 depth = 0x00;
				 depth2 = 900;

			}
			else {
				depth = 0xFF;//depth%256;
			}
            // Draw a grayscale image of the depth:
            // B,G,R are all set to depth%256, alpha set to 1.
			
			
			for (int i = 0; i < 3; ++i) { 
			 bufferdepth[index] = (char) depth;
			 index++;
			}

            bufferdepth[index] = (char) 0xff;

			bufferdepth2[index2] = (double) depth2;
		
			cvmSet(Depth2,index2/widthdepth,index2%widthdepth,bufferdepth2[index2]);
		
		}		
}

//............................................Edge detection..............................................................\\

bool CannyThreshold() {
	
	//Mat imgC(imgThresh);
	blur( imagegrey, imagegrey, Size(3,3) );
	Canny( imagegrey, detected_edges, lowThreshold, highThreshold, kernel_size );
	edges1 = Scalar::all(0);
	imagegrey.copyTo(edges1, detected_edges);

	return true;
 }

//.......................................Edge detection for motion.........................................................\\

bool Canny4Motion() {

	Mat img_diff_grey;
	cvtColor(img_diff, img_diff_grey, CV_RGB2GRAY);
	blur( img_diff_grey, img_diff_grey, Size(3,3) );
	Canny(img_diff_grey, detected_edges2, lowThreshold, highThreshold, kernel_size );
	edges2= Scalar::all(0);
	img_diff_grey.copyTo(edges2, detected_edges2);

	return true;
 }

//.......................................Edge detection for Depth...........................................................\\

bool Canny4Depth() {
	
	blur( Depth, Depth, Size(3,3) );
	Canny( Depth, detected, lowThreshold, highThreshold, kernel_size );
	edges3 = Scalar::all(0);
	detected.copyTo(edges3, imageRGBmotion);
	return true;
 }

//...................................make countours around objects in RGB frame.............................................\\

bool makeContours() {

	if(!CannyThreshold()) exit(9);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours( edges1, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	

	drawing = Mat::zeros( edges1.size(), CV_8UC3 );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	//int sumx = 0;
	//int sumy = 0;
	  
	//for( int i = 0; i < contours.size(); i++ ) { 
	//	 
	//	approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
 //       minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
	//	
	//	sumx = sumx+center[i].x;
	//	sumy = sumy+center[i].y;

 //    }

	 for( int i = 0; i < (int) contours.size(); i++ ) {
	   approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
     }

  //if(center.size() != 0) {
	 // center_cany_x = sumx/(center.size());
	 // center_cany_y = sumy/(center.size());
	 // draw = true;
	 // circle( drawing, Point(center_cany_x,center_cany_y), 100, cvScalar(0,0,255), 1, 8, 0 );
	 // //rectangle( drawing, boundRect[idBiggest].tl(), boundRect[idBiggest].br(), cvScalar(255,0,255), 1, 8, 0 );
  //}

  //else { 
	 // draw = false; 
  //}
 
  for( int i = 0; i < (int)contours.size(); i++ ) { 
	  drawContours( drawing, contours, i, cvScalar(0,0,255), 1, 8, hierarchy, 0, Point() );
	  if(boundRect[i].area() >= 9000) {
		   rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), cvScalar(255,0,0), 3, 8, 0 );
	  } 
  }
		return true;
}

//...................................make countours around objects in Depth frame............................................\\

bool makeContoursDepth() {

	if(!Canny4Depth()) exit(10);
	
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours( edges3, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	

	drawing2 = Mat::zeros( edges3.size(), CV_64FC4 );
	vector<vector<Point> > contours_poly( contours.size() );
	vector<Rect> boundRect( contours.size() );
	vector<Point2f>center( contours.size() );
	vector<float>radius( contours.size() );

	int count=0;
	double* tempArray = new double[(int) contours.size()];
	double* tempArray2 = new double[(int) contours.size()];
	
	for( int i = 0; i < (int) contours.size(); i++ ) {

	  drawContours( drawing2, contours, i, cvScalar(0,0,255), 3, 8, hierarchy, 0, Point() );
	  
	   approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
	  //CvMat cvMat = drawing2;

	   if(boundRect[i].area() >= 3000 ) {
		  
		   rectangle( drawing2, boundRect[i].tl(), boundRect[i].br(), cvScalar(255,0,0), 3, 8, 0 );

	        } 
		 }


		return true;
}

//............................................Detect Motion..................................................................\\

bool motiondetection() {	
	
	
	absdiff(img_curr, img_prev, img_diff);
	threshold(img_diff, img_diff, 100, 255, THRESH_BINARY);
	
	double beta = 0.5;
	addWeighted(img_diff, 0.5, imageRGBmotion, beta, 0.0, cumulative);
	
	
	if (! makeContours()) exit(8);

	if(draw){
		cvCircle(motion,cvPoint(center_cany_x,center_cany_y),100,cvScalar(0,255,0),2);
	}
		
	return true;
}

//............................................Detect Faces....................................................................\\

bool faceDetection( Mat frame ) {
  
  std::vector<Rect> faces;
  Mat frame_gray;

  cvtColor( frame, frame_gray, CV_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  if( !face_cascade.load( body_cascade_name ) ){ printf("--(!)Error loading\n"); exit(-3); };

  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

  for( int i = 0; i < (int) faces.size(); i++ )
  {
    Point center( faces[i].x +(int) faces[i].width/2, faces[i].y + (int) faces[i].height/2 );
    ellipse( frame, center, Size( (int) faces[i].width/2, (int)faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( faces[i] );
    
  }

	return true;
}

//.......................................................color tracking.......................................................\\

bool trackColorObject() {

	cvZero(imgTracking);
	cvCvtColor(motion, imgHSV, CV_RGB2HSV);
    cvInRangeS(imgHSV, cvScalar(lowerH,lowerS,lowerV), cvScalar(upperH,upperS,upperV), imgThresh);
	cvSmooth(imgThresh, imgThresh, CV_GAUSSIAN,3,3);
     //Calculate the moments of 'imgThresh'
    CvMoments *moments = (CvMoments*)malloc(sizeof(CvMoments));
    cvMoments(imgThresh, moments, 1);
	
    double moment10 = cvGetSpatialMoment(moments, 1, 0);
    double moment01 = cvGetSpatialMoment(moments, 0, 1);
    double area = cvGetCentralMoment(moments, 0, 0);
	

     //if the area<2000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
    if(area>2000){
         //calculate the position of the ball
        double posXd = (moment10/area);
        double posYd = (moment01/area); 
		int posX = (int)posXd;
		int posY = (int)posYd;
		int radius = (int)sqrt(area/3.142);
        
       if(lastX>=0 && lastY>=0 && posX>=0 && posY>=0)
        {
          
			cvCircle(imgTracking,cvPoint(posX,posY),radius,cvScalar(0,255,0),2);
			cvLine(imgTracking,cvPoint(posX+radius,posY-radius),cvPoint(posX+radius,posY+radius),cvScalar(0,255,0),2);
			cvLine(imgTracking,cvPoint(posX+radius,posY+radius),cvPoint(posX-radius,posY+radius),cvScalar(0,255,0),2);
			cvLine(imgTracking,cvPoint(posX-radius,posY+radius),cvPoint(posX-radius,posY-radius),cvScalar(0,255,0),2);
			cvLine(imgTracking,cvPoint(posX-radius,posY-radius),cvPoint(posX+radius,posY-radius),cvScalar(0,255,0),2);
			
			
        }

         lastX = posX;
        lastY = posY;
    }
	cvAdd(imgTracking, imageRGB, RGBout);
	free(moments); 
	return true;
}

//.......................................................Buuble sort..........................................................\\

void bubbleSort(double* a, int N) {

 bool swap = true; 
 double temp; 
  while (swap) { 
	 swap = false; 
	 for (int i=0; i<N; i++) { 
		 if (a[i] > a[i+1]) { 
			 swap = true; 
			 temp = a[i]; 
			 a[i] = a[i+1]; 
			 a[i+1] = temp; 
			} 
		} 
	} 
}

//===================================================================================================================================
void init()
{
	glShadeModel(GL_SMOOTH);     // Set the shading model to smooth 
    glClearColor(0, 0, 0, 0.0f);    // Clear the Color
    // Clear the Color and Depth Buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
    glClearDepth(1.0f);          // Set the Depth buffer value (ranges[0,1])
    glEnable(GL_DEPTH_TEST);  // Enable Depth test
    glDepthFunc(GL_LEQUAL);   // If two objects on the same coordinate 
                                            // show the first drawn
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
 
}


void drawings() {


   double minX = ExtendedPath3D.at(0);
   double maxX = ExtendedPath3D.at(ExtendedPath3D.size()-3);
   double minY = ExtendedPath3D.at(1);
   double maxY = ExtendedPath3D.at(ExtendedPath3D.size()-2);

	for(unsigned int i = 0;  i <= ExtendedPath2D.size()-3; i = i + 3) {

		if(ExtendedPath3D.at(i) < minX) minX = ExtendedPath3D.at(i);
		if(ExtendedPath3D.at(i+1) < minY) minY = ExtendedPath3D.at(i+1);
		if(ExtendedPath3D.at(i) > maxX) maxX = ExtendedPath3D.at(i);
		if(ExtendedPath3D.at(i+1) > maxY) maxY = ExtendedPath3D.at(i+1);

	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();   // It is important to push the Matrix before 
                                 // calling glRotatef and glTranslatef
            glRotatef(rotX, 1.0f, 0.0f, 0.0f);            // Rotate on x
            glRotatef(rotY, 0.0f, 1.0f, 0.0f);            // Rotate on y
            glRotatef(rotZ, 0.0f, 0.0f, 1.0f);            // Rotate on z


	if (rotation) {
           rotX += 0.2f;
           rotY += 0.2f;
           rotZ += 0.2f;
        }

	glTranslatef(X, Y, Z);
   // glMatrixMode(GL_MODELVIEW);
   // // clear the drawing buffer.
   //glClear(GL_COLOR_BUFFER_BIT);
   //glLoadIdentity();
   //glTranslatef(0,0.0,-15.5);
   //glRotatef(xRotated,1.0,0.0,0.0);
   // // rotation about Y axis
   //glRotatef(yRotated,0.0,1.0,0.0);
   // // rotation about Z axis
   //glRotatef(zRotated,0.0,0.0,1.0);
   double x1;
   double x2;
   double y1;
   double z1;
   double y2;
   double z2;

  // double startX = ((ExtendedPath3D.at(0) - minX) * (5 - (-0)) / (maxX - minX)) + (-0);
   //double startY = ((ExtendedPath3D.at(1) - minY) * (-5 - (0)) / (maxY - minY)) + (0);
  // double startZ = ((ExtendedPath3D.at(2) - 900) * (-5 - (0)) / (3000 - 900)) + (0);
   double endX = ((ExtendedPath3D.at(ExtendedPath3D.size()-3) - minX) * (11 - (-11)) / (maxX - minX)) + (-11);
   double endY = ((ExtendedPath3D.at(ExtendedPath3D.size()-2) - minY) * (-11 - (11)) / (maxY - minY)) + (11);
   double endZ = ((ExtendedPath3D.at(ExtendedPath3D.size()-1) - 900) * (-5 - (0)) / (3000 - 900)) + (0);
   
 
  for(unsigned int k = 0; k<=ExtendedPath3D.size()-6; k = k + 3 ) {	
   
	  
	
		  x1 = ((ExtendedPath3D.at(k) - minX) * (22 - (0)) / (maxX - minX)) + (0);
		  y1 = ((ExtendedPath3D.at(k+1) - minY) * (22 - (0)) / (maxY - minY)) + (0);
		  z1 = ((ExtendedPath3D.at(k+2) - 900) * (-10 - (0)) / (3000 - 900)) + (0);
		  x2 = ((ExtendedPath3D.at(k+3) - minX) * (22 - (0)) / (maxX - minX)) + (0);
		  y2 = ((ExtendedPath3D.at(k+4) - minY) * (22 - (0)) / (maxY - minY)) + (0);
		  z2 = ((ExtendedPath3D.at(k+5) - 900) * (-10 - (0)) / (3000 - 900)) + (0);

		  //		  x1 = ((ExtendedPath3D.at(k) -0) * (22 - (0)) / (maxX - 0)) + (0);
		  //y1 = ((ExtendedPath3D.at(k+1) - 0) * (22 - (0)) / (maxY - 0)) + (0);
		  //z1 = ((ExtendedPath3D.at(k+2) - 900) * (10 - (0)) / (3000 - 900)) + (0);
		  //x2 = ((ExtendedPath3D.at(k+3) - 0) * (22 - (0)) / (maxX - 0)) + (0);
		  //y2 = ((ExtendedPath3D.at(k+4) - 0) * (22 - (0)) / (maxY - 0)) + (0);
		  //z2 = ((ExtendedPath3D.at(k+5) - 900) * (10 - (0)) / (3000 - 900)) + (0);
	  

   glBegin(GL_LINES);
     // Draw The Cube Using quads
		glColor3f(1.0f,0.5f,0.0f);     // Color Blue
		glVertex3d( (x1-11)/2, -(y1-11)/2,(z1)/2);    // Top Right Of The Quad (Top)
		glVertex3d( (x2-11)/2, -(y2-11)/2,(z2)/2);

		 glEnd(); 

		 glBegin(GL_LINES);
                glColor3f(0.0f, 1.0f, 0.0f);                // Green for x axis
                glVertex3f(-3.0f, 0.0f, 0.0f);
                glVertex3f(3.0f, 0.0f, 0.0f);
                glColor3f(1.0f, 0.0f, 0.0f);                // Red for y axis
                glVertex3f(0.0f, -3.0f, 0.0f);
                glVertex3f(0.0f, 3.0f, 0.0f);
                glColor3f(0.0f, 0.0f, 1.0f);                // Blue for z axis
                glVertex3f(0.0f, 0.0f, -3.0f);
                glVertex3f(0.0f, 0.0f, 3.0f);
          glEnd();

		 glBegin(GL_POINTS);
			//glColor3f(1.0f,0.5f,0.0f);     // Color Blue
			//glVertex3d( startX, startY,startZ);    // Top Right Of The Quad (Top)
			glColor3f(2.0f,2.0f,2.0f);
			glVertex3d( endX/2+1, endY/2+1,endZ/2+1);

		 glEnd();

		 glFlush();
  }
  		 glutPostRedisplay();           // Redraw the scene
         glPopMatrix();                   // Don't forget to pop the Matrix
         glutSwapBuffers();
  //glutLeaveMainLoop();
}


void keyboard(byte key, int x, int y) {
            switch (key)
            {
                // x,X,y,Y,z,Z uses the glRotatef() function
                case 120:    // x             // Rotates screen on x axis 
                    rotX -= 2.0f;
                    break;
                case 88:    // X            // Opposite way 
                    rotX += 2.0f;
                    break;
                case 121:    // y            // Rotates screen on y axis
                    rotY -= 2.0f;
                    break;
                case 89:    // Y            // Opposite way
                    rotY += 2.0f;
                    break;
                case 122:    // z            // Rotates screen on z axis
                    rotZ -= 2.0f;
                    break;
                case 90:    // Z            // Opposite way
                    rotZ += 2.0f;
                    break;

                // j,J,k,K,l,L uses the gluLookAt function for navigation
                case 106:   // j
                    rotLx -= 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 74:    // J
                    rotLx += 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 107:   // k
                    rotLy -= 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 75:    // K
                    rotLy += 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 108: // (l) It has a special case when the rotLZ becomes 
                          // less than -15 the screen is viewed from the opposite side
                    // therefore this if statement below does not allow 
                    // rotLz be less than -15
                    if (rotLz + 14 >= 0)
                        rotLz -= 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 76:    // L
                    rotLz += 2.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0 + rotLz, 
                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
                    break;
                case 98:    // b        // Rotates on x axis by -90 degree
                    rotX -= 90.0f;
                    break;
                case 66:    // B        // Rotates on y axis by 90 degree
                    rotX += 90.0f;
                    break;
                case 110:    // n        // Rotates on y axis by -90 degree
                    rotY -= 90.0f;
                    break;
                case 78:    // N        // Rotates on y axis by 90 degree
                    rotY += 90.0f;
                    break;
                case 109:    // m        // Rotates on z axis by -90 degree
                    rotZ -= 90.0f;
                    break;
                case 77:    // M        // Rotates on z axis by 90 degree
                    rotZ += 90.0f;
                    break;
                case 111:    // o        // Resets all parameters
                case 80:    // O        // Displays the cube in the starting position
                    rotation = false;
                    X = Y = 0.0f;
                    Z = 0.0f;
                    rotX = 0.0f;
                    rotY = 0.0f;
                    rotZ = 0.0f;
                    rotLx = 0.0f;
                    rotLy = 0.0f;
                    rotLz = 0.0f;
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(rotLx, rotLy, 15.0f + rotLz, 
                        0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
                    break;
            }
            glutPostRedisplay();    // Redraw the scene
        }


void reshape(int w, int h)
{
    //if (y == 0 || x == 0) return;  //Nothing is visible then, so return
    ////Set a new projection matrix
    //glMatrixMode(GL_PROJECTION);  
    //glLoadIdentity();
    ////Angle of view:30 degrees
    ////Near clipping plane distance: 0.5
    ////Far clipping plane distance: 20.0
    // 
    //gluPerspective(-50.0,(GLdouble)x/(GLdouble)y,0.5,20.0);
    //glMatrixMode(GL_MODELVIEW);
    //glViewport(0,0,x,y);  //Use the whole window for rendering


   glViewport(0, 0, w, h);                // Set the viewport
   glMatrixMode(GL_PROJECTION);        // Set the Matrix mode
   glLoadIdentity();
   gluPerspective(75, (GLfloat)w / (GLfloat)h, 0.10, 500.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   gluLookAt(rotLx, rotLy, 15.0f + rotLz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

}


static void specialKey(int key, int x, int y)
        {
            // Check which key is pressed
            switch (key)
            {
                case GLUT_KEY_LEFT:    // Rotate on x axis
                    X -= 2.0f;
                    break;
                case GLUT_KEY_RIGHT:    // Rotate on x axis (opposite)
                    X += 2.0f;
                    break;
                case GLUT_KEY_UP:        // Rotate on y axis 
                    Y += 2.0f;
                    break;
                case GLUT_KEY_DOWN:    // Rotate on y axis (opposite)
                    Y -= 2.0f;
                    break;
                case GLUT_KEY_PAGE_UP:  // Rotate on z axis
                    Z -= 2.0f;
                    break;
                case GLUT_KEY_PAGE_DOWN:// Rotate on z axis (opposite)
                    Z += 2.0f;
                    break;
                case GLUT_KEY_F1:      // Enable/Disable coordinate lines
                    lines = !lines;
                    break;
                case GLUT_KEY_F2:      // Enable/Disable automatic rotation
                    rotation = !rotation;
                    break;
                default:
                    break;
            }
            glutPostRedisplay();        // Redraw the scene
        }

        // Capture the mouse click event 
//void processMouseActiveMotion(int button, int state, int x, int y)
//        {
//            mousePressed = button;          // Capture which mouse button is down
//            old_x = x;                      // Capture the x value
//            old_y = y;                      // Capture the y value
//        }
//
//        // Translate the x,y windows coordinates to OpenGL coordinates
//void processMouse(int x, int y) {
//            if ((mousePressed == 0))    // If left mouse button is pressed
//            {
//                X = (x - old_x) / 15;       // I did divide by 15 to adjust 
//                                            // for a nice translation 
//                Y = -(y - old_y) / 15;
//            }
//
//            glutPostRedisplay();
//        }

        // Get the mouse wheel direction
static void processMouseWheel(int wheel, int direction, int x, int y)
        {
            Z += direction;  // Adjust the Z value 
            glutPostRedisplay();
        }


//====================================================================================================================================

double CubicInterpolate(double y0, double y1, double y2, double y3, double mu) {
  
  double a0,a1,a2,a3,mu2;

   mu2 = mu*mu;
   a0 = y3 - y2 - y0 + y1;
   a1 = y0 - y1 - a0;
   a2 = y2 - y0;
   a3 = y1;

   return(a0*mu*mu2+a1*mu2+a2*mu+a3);
}
 

void extendPath() {
	int x0;
	int x1;
	int x2;
	int x3;

	int y0;
	int y1;
	int y2;
	int y3;
	ExtendedPath2D.clear();
	
	for(unsigned int i = 0; i <= path2D.size() - 4; i = i+2) {

		if(path2D.size() == 4) {
			x0 = path2D.at(i);
			y0 = path2D.at(i+1);

			x1 = path2D.at(i);
			y1 = path2D.at(i+1);

			x2 = path2D.at(i+2);
			y2 = path2D.at(i+3);

			x3 = path2D.at(i+2);
			y3 = path2D.at(i+3);
		}

		else if( i == 0) {

			x0 = path2D.at(i);
			y0 = path2D.at(i+1);
			
			x1 = path2D.at(i);
			x2 = path2D.at(i+2);
			x3 = path2D.at(i+4);

			y1 = path2D.at(i+1);
			y2 = path2D.at(i+3);
			y3 = path2D.at(i+5);

		}
		else if (i == path2D.size() - 4) {
			
			x0 = path2D.at(i-2);
			y0 = path2D.at(i-1);

			x1 = path2D.at(i);
			x2 = path2D.at(i+2);
			
			y1 = path2D.at(i+1);
			y2 = path2D.at(i+3);

			x3 = path2D.at(i+2);
			y3 = path2D.at(i+3);
		
		}
		else {

			x0 = path2D.at(i-2);
			y0 = path2D.at(i-1);

			x1 = path2D.at(i);
			x2 = path2D.at(i+2);
			x3 = path2D.at(i+4);

			y1 = path2D.at(i+1);
			y2 = path2D.at(i+3);
			y3 = path2D.at(i+5);
			
		}
	
		for(unsigned int j = 0; j<=10; j++) {

			ExtendedPath2D.push_back( (int) CubicInterpolate(x0, x1, x2, x3, j*0.1 ) );
			ExtendedPath2D.push_back( (int) CubicInterpolate(y0, y1, y2, y3, j*0.1 ) );
		}

	}

}


void extendPath3D() {

	double x0;
	double x1;
	double x2;
	double x3;

	double y0;
	double y1;
	double y2;
	double y3;

	double z0;
	double z1;
	double z2;
	double z3;
	
	ExtendedPath3D.clear();
	
	for(unsigned int i = 0; i <= path.size() - 6; i = i+3) {

		if(path.size() == 6) {
			x0 = path.at(i);
			y0 = path.at(i+1);
			z0 = path.at(i+2);

			x1 = path.at(i);
			y1 = path.at(i+1);
			z1 = path.at(i+2);

			x2 = path.at(i+3);
			y2 = path.at(i+4);
			z2 = path.at(i+5);

			x3 = path.at(i+3);
			y3 = path.at(i+4);
			z3 = path.at(i+5);
		}

		else if( i == 0) {

			x0 = path.at(i);
			y0 = path.at(i+1);
			z0 = path.at(i+2);

			x1 = path.at(i);
			x2 = path.at(i+3);
			x3 = path.at(i+6);

			y1 = path.at(i+1);
			y2 = path.at(i+4);
			y3 = path.at(i+7);

			z1 = path.at(i+2);
			z2 = path.at(i+5);
			z3 = path.at(i+8);

		}
		else if (i == path.size() - 6) {
			
			x0 = path.at(i-3);
			y0 = path.at(i-2);
			z0 = path.at(i-1);

			x1 = path.at(i);
			x2 = path.at(i+3);
			
			y1 = path.at(i+1);
			y2 = path.at(i+4);

			x3 = path.at(i+3);
			y3 = path.at(i+4);

			z1 = path.at(i+2);
			z2 = path.at(i+5);
			z3 = path.at(i+5);
		
		}
		else {

			x0 = path.at(i-3);
			y0 = path.at(i-2);
			z0 = path.at(i-1);

			x1 = path.at(i);
			x2 = path.at(i+3);
			x3 = path.at(i+6);

			y1 = path.at(i+1);
			y2 = path.at(i+4);
			y3 = path.at(i+7);

			z1 = path.at(i+2);
			z2 = path.at(i+5);
			z3 = path.at(i+8);
			
		}
	
		for(unsigned int j = 0; j <= 10; j++) {

			ExtendedPath3D.push_back(  CubicInterpolate(x0, x1, x2, x3, j*0.1) );
			ExtendedPath3D.push_back(  CubicInterpolate(y0, y1, y2, y3, j*0.1) );
			ExtendedPath3D.push_back(  CubicInterpolate(z0, z1, z2, z3, j*0.1) );
		}

	}

}


void ExtendedPath3DAxisTransformation() {//bx, bz postion of copter on frame

	double b1; 
	double b2;


	double x;
	double y;
	double z;

	newExtendedPath3D.clear();

	for(unsigned int i = 0; i <= ExtendedPath3D.size()-3; i=i+3) {

		z = ExtendedPath3D.at(i+2);
		b1 = 0.54861878662 * z;
		x = ExtendedPath3D.at(i) - b1;

		b2 = 0.39895954597 * z;
		y = b2 - ExtendedPath3D.at(i+1);

		newExtendedPath3D.push_back(x);
		newExtendedPath3D.push_back(y);
		newExtendedPath3D.push_back(z);



		//2D only X=Z
		//GlobalExtendedPath3D.push_back(x+bx);
		//GlobalExtendedPath3D.push_back(z+bz);

	}

}


//void AstarextendPath() {
//	int x0;
//	int x1;
//	int x2;
//	int x3;
//
//	int y0;
//	int y1;
//	int y2;
//	int y3;
//	ExtendedAstar.clear();
//	
//	for(unsigned int i = 0; i <=AstarPath.size() - 4; i = i+2) {
//
//		if(AstarPath.size() == 4) {
//			x0 = AstarPath.at(i);
//			y0 = AstarPath.at(i+1);
//
//			x1 = AstarPath.at(i);
//			y1 = AstarPath.at(i+1);
//
//			x2 = AstarPath.at(i+2);
//			y2 = AstarPath.at(i+3);
//
//			x3 = AstarPath.at(i+2);
//			y3 = AstarPath.at(i+3);
//		}
//
//		else if( i == 0) {
//
//			x0 = AstarPath.at(i);
//			y0 = AstarPath.at(i+1);
//			
//			x1 = AstarPath.at(i);
//			x2 = AstarPath.at(i+2);
//			x3 = AstarPath.at(i+4);
//
//			y1 = AstarPath.at(i+1);
//			y2 = AstarPath.at(i+3);
//			y3 = AstarPath.at(i+5);
//
//		}
//		else if (i == AstarPath.size() - 4) {
//			
//			x0 = AstarPath.at(i-2);
//			y0 = AstarPath.at(i-1);
//
//			x1 = AstarPath.at(i);
//			x2 = AstarPath.at(i+2);
//			
//			y1 = AstarPath.at(i+1);
//			y2 = AstarPath.at(i+3);
//
//			x3 = AstarPath.at(i+2);
//			y3 = AstarPath.at(i+3);
//		
//		}
//		else {
//
//			x0 = AstarPath.at(i-2);
//			y0 = AstarPath.at(i-1);
//
//			x1 = AstarPath.at(i);
//			x2 = AstarPath.at(i+2);
//			x3 = AstarPath.at(i+4);
//
//			y1 = AstarPath.at(i+1);
//			y2 = AstarPath.at(i+3);
//			y3 = AstarPath.at(i+5);
//			
//		}
//	
//		for(unsigned int j = 0; j<=10; j++) {
//
//			ExtendedAstar.push_back( (int) CubicInterpolate(x0, x1, x2, x3, j*0.1 ) );
//			ExtendedAstar.push_back( (int) CubicInterpolate(y0, y1, y2, y3, j*0.1 ) );
//		}
//
//	}
//
//}


//.......................................................Blobe detection.......................................................\\

bool BlobDetection(IplImage* img) {
	
	
	 
	cvCvtColor(img,originalThr,CV_RGBA2GRAY);
	cvThreshold( originalThr, originalThr, 0, 255, CV_THRESH_BINARY );
	
	// find non-white blobs in thresholded image
	blobs = CBlobResult( originalThr, NULL, 0x00 );
	
	// exclude the ones smaller than param2 value
	//blobs.Filter( blobs, B_EXCLUDE, CBlobGetArea(), B_LESS,975 );
	
	CBlob biggestBlob;
	CBlobGetMean getMeanColor(originalThr );
	double meanGray;
	
	meanGray = getMeanColor( biggestBlob );
	
	// display filtered blobs
	//cvMerge( originalThr, originalT, original, NULL, displayedImage );

	numberOfObstacles = blobs.GetNumBlobs();
	distArray.resize((int) blobs.GetNumBlobs());
	obstacle2DPosArray.resize((int) blobs.GetNumBlobs() *4);
	obstaclePosArray.resize((int) blobs.GetNumBlobs() *4);

	bool isMidRect = false;
	for (int i = 0; i < blobs.GetNumBlobs(); i++ ) {
		
		currentBlob = blobs.GetBlob(i);
		
		cv::Rect rect(0,0,0,0);
		
		rect =  currentBlob->GetBoundingBox();
		//float area = rect.area();
		
		//currentBlob->FillBlob( displayedImage, CV_RGB(0,0,255));
        cvRectangle( displayedImage, rect.tl(), rect.br() ,  cvScalar(255, 0,0,0 ),3 );
		
		if(rect.tl().x <= widthdepth/2 && rect.tl().y <=heightdepth/2  && rect.br().x >= widthdepth/2 && rect.br().y >=heightdepth/2 ) {
			isMidRect = true;
		}

		 cv::Point tl = rect.tl();
			   int xI = tl.x;
			   int yI = tl.y;

		   double sum = 0;
		   cv::Mat frame2(originalThr);
		   int counter = 0;
		   unsigned char *input = (unsigned char*)(frame2.data);
		   for(int u=0; u<rect.height; u++) {
			   for(int j=0; j<rect.width; j++) {
				   if( input[(frame2.cols*(yI+u)) + (xI+j)] != 0) {
					   sum = sum + cvmGet(Depth2, yI+u ,xI+j);
					   counter++;
				   }
				}
			}
		double distance;
		if (counter == 0) distance = 3000;
		else distance = sum / counter;

		
	    if(isMidRect) midObjDistAvg = distance; 
		double pixelWidth = (distance * 1.08591139928) / widthdepth;
		double pixelHeight = (distance * 0.78782095123) / heightdepth;

		double RealX =  xI * pixelWidth;
		double RealY =  yI * pixelHeight;
		double RealWidth =  (rect.width) *  pixelWidth;
		double RealHeight = (rect.height) *  pixelHeight;

			distArray.at(i) = distance;
			obstacle2DPosArray.at(4*i) = RealX;
			obstacle2DPosArray.at(4*i + 1) = RealY;
			obstacle2DPosArray.at(4*i + 2) =  RealWidth;
			obstacle2DPosArray.at(4*i + 3) = RealHeight;

			obstaclePosArray.at(4*i) = xI;
			obstaclePosArray.at(4*i + 1) = yI;
			obstaclePosArray.at(4*i + 2) =  rect.width;
			obstaclePosArray.at(4*i + 3) = rect.height;
		
	}

	// bubbleSort(distArray, blobs.GetNumBlobs());
	if(distArray.size() != 0) {	
			 nearestDst = distArray.at(0);
			 double max = distArray.at(0);
			 
		  for(int k = 0; k< blobs.GetNumBlobs(); k++) {
			  if(distArray.at(k) < nearestDst) nearestDst = distArray.at(k);
			  if(distArray.at(k) > max) max = distArray.at(k);
			 }

	}		  //exit(obstacle2DPosArray[11]);
	cvAdd(img, displayedImage, displayedImage);
	//exit(blobs.GetNumBlobs());
	//exit( blobs.GetNumBlobs());
	
	return true;
}

//..................................................find Element Index In Array.................................................\\

int findElementIndexInArray(double* a, int N, double x) {

	int i;
	for(i = 0; i<N; i++) {
		if(a[i] ==  x) {
			return i;
		}
	}
		return -1;
}

//....................................................find Safe Area.............................................................\\

void findSafeArea(double nearestDist) {
	//nearestDist = 1000;
	//the kinect can see fom a distance tat is equal to 900 mm, so at 800 mm the copter(1000x300 mm) should be...
	double pixelWidth = (nearestDist * 1.08591139928) / widthdepth; //2dtan(57/2) / 640
	double pixelHeight = (nearestDist * 0.78782095123) / heightdepth; //2dtan(43/2) / 480
	
	int copterwidth = 150;//8000/pixelWidth;
	int copterHeight = 50;//230/pixelHeight;

	cv::Mat copterArea = cv::Mat::zeros((int)copterHeight, (int)copterwidth, CV_8UC1);
	int result;//cv::Mat result = cv::Mat::zeros(50, 150, CV_8UC1);
	cv::Mat frame(originalThr);
	//vector<Rect> safeRects( heightdepth - 50 );
	//FILE *fp;
	//char output[]="C:\\sdl\\output.txt";
	//fp=fopen(output,"w");

	int u = 0;
	int v = 0;
	int du = 0;
	int t = 0;
	int Unew = 0;
	int Vnew = 0;
	int dv = -5;
	int Umax = (frame.cols - Xold) - (copterArea.cols/2);
	int Umin = (-Xold)  + (copterArea.cols/2);
	int Vmax = Yold - (copterArea.rows/2);
	int Vmin = -20;
	double bx = currLocation.at(0);
	double bz = currLocation.at(1);
	double Xgf;
	double xCop;

	//====
	bool found = false;
	for(unsigned int i = 0; i <= AstarPath3D.size() - 2; i = i + 2) {

		if(AstarPath3D.at(i+1) >= bz+3000) {
			Xgf = AstarPath3D.at(i);
			found = true;
			break;
		}

	}
	if(found == false){
		Xgf =  AstarPath3D.at(AstarPath3D.size()-2);
		//return; //make it land;
	}
	xCop = 1645.85635988 + (bx - Xgf); //3000 tan57.5/2
	Xold = (int) (xCop / 5.14330112463); //(2 * 3000 * tan57.5/2)  / 640

	if(Xold < copterwidth/2) Xold = copterwidth/2;
	else if(Xold > widthdepth -  copterwidth/2) Xold = widthdepth -  copterwidth/2;

	unsigned char *input = (unsigned char*)(frame.data);
	unsigned char *input2 = (unsigned char*)(copterArea.data);

	

	for(int i = 0; i < ((frame.cols)*(frame.cols)); i=i+5){
		
		 if ( ((Umin < u) && (Umax > u))   &&    ((Vmin <= v) && (Vmax > v)) ) {
				
			 bool flagValid = true; 
			for(int k=0; k<copterArea.rows; k++) {
				for(int l=0; l<copterArea.cols; l++) {

					Unew = u - copterArea.cols/2;
					Vnew = v + copterArea.rows/2;
					result = input[((frame.cols) * ((frame.rows-(Yold+Vnew))+k)) + ((Xold+Unew)+l)] + input2[copterArea.cols * k + l];
					
					if( result != 0) {
						flagValid = false;
						break;
					}
                }
				if(flagValid == false) break;
            }
		
			if(flagValid == true) {
				  kalmanInputX = Xold+Unew;
				  kalmanInputY = Yold+Vnew;

				  cv::Point tl2((Xold+Unew), frame.rows-(Yold+Vnew));
				  cv::Point br2((Xold+Unew)+(int)copterwidth, (frame.rows-(Yold+Vnew))+(int)copterHeight );
				  cvRectangle( displayedImage, tl2, br2 ,  cvScalar(0, 255, 255 ),1 );

				  cv::Point tl((int)(updatedX), frame.rows-((int)updatedY));
				  cv::Point br(((int)updatedX)+(int)copterwidth, (frame.rows-((int)updatedY))+(int)copterHeight );
				  cvRectangle( displayedImage, tl, br ,  cvScalar(255, 255, 255 ),1 );
				  
				  if(! drawTunnel3D(tl,  br)) exit(99);
				 
				  else {
					  extendPath();
					  extendPath3D();
					  ExtendedPath3DAxisTransformation();
					  for(unsigned int k = 0; k<=ExtendedPath2D.size()-4; k = k + 2 ) {
						cvLine(displayedImage, cv::Point (ExtendedPath2D.at(k), ExtendedPath2D.at(k+1)), cv::Point (ExtendedPath2D.at(k+2), ExtendedPath2D.at(k+3)), cvScalar(0, 0, 255), 2, 8, 0);	
					}
					
					  for(unsigned int k = 0; k<=path2D.size()-4; k = k + 2 ) {
						cvRectangle(displayedImage, cv::Point (path2D.at(k)-(int)copterwidth/2, path2D.at(k+1)-(int)copterHeight/2), cv::Point (path2D.at(k)+(int)copterwidth/2, path2D.at(k+1)+(int)copterHeight/2), cvScalar(0, 0, 255), 1);	
						cvLine(displayedImage, cv::Point (path2D.at(k), path2D.at(k+1)), cv::Point (path2D.at(k+2), path2D.at(k+3)), cvScalar(0, 255, 250), 2, 8, 0);	
					}

				  }
				  break;
			}
		 }
			if( (u == v) || ((u < 0) && (u == -v)) || ((u > 0) && (u == 5-v)) ) {
				t = du;
				du = -dv;
				dv = t;
			}

        u = u + du;
		v = v + dv;
	}
	 //fclose(fp);
	//free(boolArray);

  if(  (Xold + u> 0) && (Xold + u < widthdepth)  &&  (150 < Yold - v) && ( Yold - v < heightdepth) ) {
		//Xold = Xold + u;
		Yold = Yold - v;
  }


 
	return;
}

//..................................................find if Element is In Array...................................................\\

bool isElementInVector(vector<node*> v, node* element) {

	for(unsigned int i = 0; i < v.size(); i++) {
		if( (v.at(i)->getXreal() == element->getXreal()) && (v.at(i)->getYreal() == element->getYreal()) && (v.at(i)->getZreal() == element->getZreal())) return true;
	}
	return false;
}


bool isElementInList(vector<node*> v, node* element) {

	for(unsigned int i = 0; i < v.size(); i++) {
		if( (v.at(i)->getX() == element->getX()) && (v.at(i)->getY() == element->getY()) ) return true;
	}
	return false;
}
//..................................................find if Element Index In Vector...............................................\\

int findElementIndexInVector(vector<node*> v, node* element) {

	unsigned int i;
	for(i = 0; i<v.size(); i++) {
		if( (v.at(i)->getXreal() == element->getXreal()) && (v.at(i)->getYreal() == element->getYreal()) && (v.at(i)->getZreal() == element->getZreal())) {
			return i;
		}
	}
		return -1;
}


int findElementIndexInList(vector<node*> v, node* element) {

	for(unsigned int i = 0; i < v.size(); i++) {
		if( (v.at(i)->getX() == element->getX()) && (v.at(i)->getY() == element->getY()) ) return i;
	}
	return -1;
}

//....................................................swap elments to end........................................................\\

void swapElemnent(vector<double> v, int index) {

	double temp;
	for (unsigned int i = index; i < v.size(); i++) {

		temp = v.at(i);
		v.at(i) = v.at(i+1);
		v.at(i+1) = temp;

	}

}

//....................................................swap elments to end........................................................\\

void swapElemnentInts(vector<int> v, int index) {

	int temp;
	for (unsigned int i = index; i < v.size(); i++) {

		temp = v.at(i) ;
		v.at(i) = v.at(i+1);
		v.at(i+1) = temp;

	}

}

//....................................................reverse vector order........................................................\\

void reverseVecOrder(vector<double> v) {

	vector<double> temp;

	for(unsigned int i = (v.size() - 1 ); i>=0; i--) {

		temp.push_back(v.at(i));
		v.pop_back();
		if(i == 0) break;
	}
	
	for(unsigned int k = 0; k<temp.size(); k++) {
		v.push_back( temp.at(k));
	}
	temp.clear();
}

//....................................................draw Tunnel in space.......................................................\\
//Dont enter unless square is there and square should not be drone if dist/3000
bool drawTunnel3D(cv::Point Etl,  cv::Point Ebr) {

	double realzCurr;
	double copterNewRealZ;
	double copterNewRealX;
	double copterNewRealY;
	int copterWidth;
	int copterHeight;

	int deltaX = 0;
	int deltaY = 0;

	double realxCurr;
	double realyCurr;
	int dx[9] = {};
	int dy[9] = {};
	//double pixelWidth;
	//double pixelHeight;

	int counter3 = 0;
	int Index;
	double minFcost;
	unsigned int index2;
	int index4;

	double avg;
	double sum;

	cv::Mat frame2(originalThr);
	int counterX = 0;
	int counterX2 = 0;
	unsigned char *input = (unsigned char*)(frame2.data);

	double diffx;
	double diffy;
	double diffz;
//=================================================================
	 node* currentNode;
	 node* newNode;
	 node* end;
	 vector<node*> open_nodes;
	 vector<node*> closed_nodes;
	 double Gcost;
	 double Hcost;
	 double ParentGcost;
	 double tempGcost;
	
//===================================================================
	 
	 //define the end point
		int midXend = Etl.x + (Ebr.x - Etl.x) / 2;
	    int midYend = Etl.y + (Ebr.y - Etl.y) / 2;


		end = new node(midXend, midYend);
		end->setZreal(3000);
		end->setXreal();
		end->setYreal();



	currentNode = new node(widthdepth/2, heightdepth/2);
	currentNode->setZreal(900);
	currentNode->setXreal();
	currentNode->setYreal();
	currentNode->setIndex(0);
	currentNode->setParentIndex(-1);
		
	realzCurr = 900;
	realxCurr = currentNode->getXreal();
	realyCurr = currentNode->getYreal();

		ParentGcost = 0;

		currentNode->setGcost(ParentGcost);
		closed_nodes.push_back(currentNode);


  bool flagDone = false;
  
  while(1) {

	    //pixelWidth = (realzCurr * 1.08591139928) / widthdepth; //2dtan(57/2) / 640
	   // pixelHeight = (realzCurr * 0.78782095123) / heightdepth; //2dtan(43/2) / 480

		copterWidth = 150;//(int) (150 / pixelWidth);
		copterHeight = 50;//(int) (50 / pixelHeight);
		

		
		deltaX = copterWidth; //(ObsMidX-copterMidXCurr) + (ObsWidth/2) + (copterWidth/2);
		deltaY = -copterHeight;//(ObsMidY-copterMidYCurr) - (ObsHeight/2) - (copterHeight/2);

	    dx[0] = 0;
	    dy[0] = 0;
		dx[1] = deltaX;
	    dy[1] = 0;
	    dx[2] = deltaX;
	    dy[2] = deltaY;
		dx[3] = 0;
	    dy[3] = deltaY;
	    dx[4] = -deltaX;
	    dy[4] = deltaY;	
		dx[5] = -deltaX;
	    dy[5] = 0;
		dx[6] = -deltaX;
	    dy[6] = -deltaY;
		dx[7] = 0;
	    dy[7] = -deltaY;
		dx[8] = deltaX;
	    dy[8] = -deltaY;

		//
		//{0, deltaX, deltaX, 0, -deltaX, -deltaX, -deltaX, 0, deltaX;};
		//{0, 0, deltaY, deltaY, deltaY, 0, -deltaY, -deltaY, -deltaY;};

		for(int k  = 0; k < 9; k++) {
					
		 if ( (((currentNode->getX()) + dx[k]) <= ((widthdepth) - (copterWidth/2))) && (((currentNode->getX()) + dx[k]) >= copterWidth/2) 
			&&    (((currentNode->getY()) + dy[k]) >= copterHeight/2) && (((currentNode->getY()) + dy[k]) <= heightdepth/1.5) ) {


			newNode = new node(((currentNode->getX()) + dx[k]), ((currentNode->getY()) + dy[k]));
//===============================================================================================================
			sum = 0;

			counterX = 0;

			for(int r = (newNode->getY() - (copterHeight/2)); r <= (newNode->getY() + copterHeight); r++) {
				for(int c = (newNode->getX() - (copterWidth/2)); c <= (newNode->getX() + copterWidth); c++) {
					if( input[(frame2.cols*r) + c] != 0) {
						sum =  sum + cvmGet(Depth2, r, c);
						counterX++;
					}
				}
			}
			if (counterX != 0) avg = sum /  counterX;
			else avg = 3000;
           
			copterNewRealZ = avg;
//===============================================================================================================

		   if((copterNewRealZ < currentNode->getZreal() )) continue;
			newNode->setZreal(copterNewRealZ);
			newNode->setXreal();
			newNode->setYreal();
			
			copterNewRealX =  newNode->getXreal();
			copterNewRealY =  newNode->getYreal();

		
				
				//UNLESS NO OTHER WAY
				
				Hcost = sqrt( ( ((end->getXreal()) - copterNewRealX) * ((end->getXreal()) - copterNewRealX) )
					+ ( ((end->getYreal()) - copterNewRealY) * ((end->getYreal()) - copterNewRealY) )
					+ ( ((end->getZreal()) - copterNewRealZ) * ((end->getZreal()) - copterNewRealZ) ) );

				if( ((midXend >= (newNode->getX() - copterWidth/2))  && (((newNode->getX() - copterWidth/2) + 150) >= midXend)) && (((newNode->getY() - copterHeight/2) <= midYend) &&  (((newNode->getY() - copterHeight/2) + 50) >= midYend)) /*&& (3000 - (newNode->getZreal()) < 150)*/ ) {
						 flagDone = true;
						 // break;
					 }
				newNode->setHcost(Hcost);

				if( (! isElementInVector(open_nodes, newNode))  &&  (! isElementInVector(closed_nodes, newNode))  ) {
					  counter3++;
					  newNode->setIndex( counter3 );

					  Gcost = (currentNode->getGcost()) 
						  + sqrt( ( (copterNewRealX - realxCurr) * (copterNewRealX - realxCurr) ) 
						  +  ( (copterNewRealY - realyCurr) * (copterNewRealY - realyCurr) ) 
						  +  ( (copterNewRealZ - realzCurr) * (copterNewRealZ - realzCurr) ) );

					
					  newNode->setGcost(Gcost);
					  newNode->setFcost();
					  newNode->setParentIndex(currentNode->getIndex());
					  open_nodes.push_back(newNode);  

					  
				}

				else if( isElementInVector(open_nodes, newNode) ) {
					//check if beter to go direct from parent of parent
					
					tempGcost =  currentNode->getGcost() + sqrt( ((realxCurr-copterNewRealX)*(realxCurr-copterNewRealX)) 
						+  ((realyCurr-copterNewRealY)*(realyCurr-copterNewRealY)) +  ((realzCurr-copterNewRealZ)*(realzCurr-copterNewRealZ)) );

					Index = findElementIndexInVector(open_nodes, newNode);
					if(Index == -1)  exit(88);
					
					if(tempGcost < (open_nodes.at(Index)->getGcost()) ) {
					  
					   open_nodes.at(Index)->setGcost(tempGcost);
					   open_nodes.at(Index)->setFcost();
					   open_nodes.at(Index)->setParentIndex(currentNode->getIndex());

					
				 }
			  }

				if(flagDone  == true){
					closed_nodes.push_back(newNode);
					break;
				}
		   }
		}

		if(flagDone  == true) {
			break;
		}


		minFcost = (open_nodes.at(0))->getFcost();
		index2 = 0;
		for(unsigned int j = 0; j<open_nodes.size(); j++) {
			if( (open_nodes.at( j))->getFcost() < minFcost ) {
				minFcost = (open_nodes.at(j))->getFcost();
				index2 = j;
			}
		  }

		currentNode = new node();
		currentNode->setX( (open_nodes.at(index2))->getX() );
		currentNode->setY( (open_nodes.at( index2))->getY() );
		currentNode->setZreal( (open_nodes.at( index2))->getZreal() );
		currentNode->setYreal( );
		currentNode->setXreal( );
		currentNode->setIndex( (open_nodes.at( index2))->getIndex() );
		currentNode->setParentIndex( (open_nodes.at( index2))->getParentIndex() );
		currentNode->setGcost( (open_nodes.at(index2))->getGcost() );
		currentNode->setHcost( (open_nodes.at( index2))->getHcost() );
		currentNode->setFcost( );

	
		//check if it was easier to go directly from start

					
			//add to closed
			closed_nodes.push_back(currentNode);

			//remove from open list
			
		
			index4 = findElementIndexInVector(open_nodes, currentNode);
			if(index4 == -1)  exit(55);
			open_nodes.erase(open_nodes.begin() + index4);
			
			realxCurr = currentNode->getXreal();
			realyCurr = currentNode->getYreal();
			realzCurr = currentNode->getZreal();

			//if(flagDone  == true) break;

			if(open_nodes.size() == 1){
				break;
			}
		}
 
	

  //calc path
  path.clear();
  path2D.clear();
 
  double x;
  double y;
  double z;
  double totalDiff;
  
  path.push_back(end->getZreal());
  path.push_back(end->getYreal());
  path.push_back(end->getXreal());
    
  path2D.push_back(end->getY());
  path2D.push_back(end->getX());
  
  unsigned int i;
  int coco = 0;
	x = closed_nodes.at(0)->getX();
	y = closed_nodes.at(0)->getY();
	z = closed_nodes.at(0)->getZreal();
	diffx = abs( (end->getX()) - x );
	diffy = abs( (end->getY()) - y );
	diffz = abs( (end->getZreal()) - z );
	totalDiff = diffx + diffy;

 for(i = 1; i<closed_nodes.size(); i++) {
	x = closed_nodes.at(i)->getX();
	y = closed_nodes.at(i)->getY();
	z = closed_nodes.at(i)->getZreal();
	diffx = abs( (end->getX()) - x );
	diffy = abs( (end->getY()) - y );
	diffz = abs( (end->getZreal()) - z );
	

	if( diffx + diffy  < totalDiff) {
		totalDiff = diffx + diffy ;
		coco = i;
	}
	  }
 
 node* tile;
 
 if(flagDone == false) tile  = closed_nodes.at(coco);
 else tile  = closed_nodes.at(closed_nodes.size() - 1);
 //tile  = closed_nodes.at(coco);
 //tile  = closed_nodes.at(closed_nodes.size() - 1);

  int parentIndex;
  unsigned int j;
  
  while(1) {

	  z = tile->getZreal();
	  y = tile->getYreal();
	  x = tile->getXreal();
	 path.push_back(z);
	 path.push_back(y);
	 path.push_back(x);
  
	 path2D.push_back(tile->getY());
	 path2D.push_back(tile->getX());
	 
	 if(tile->getParentIndex() == -1) break;

	/* if(x == start->getXreal() && y == start->getYreal() && z == start->getZreal()) {
		 break;
	 }*/


	 parentIndex = tile->getParentIndex();
	 j = 0;
	 for(j = 0; j<closed_nodes.size(); j++) {
			if( (closed_nodes.at(j))->getIndex() ==  parentIndex ) {
				break;
			}
	 }
	 
	tile = closed_nodes.at(j);
	
  }


  //=================================
  	vector<double> temp;
	
	for(unsigned int i = (path.size() - 1 ); i>=0; i--) {
		temp.push_back(path.at(i));
		path.pop_back();
		if(i == 0) break;
	}
	//int cnt = 2;
	for(unsigned int k = 0; k<temp.size(); k = k + 3 ) {
		path.push_back( temp.at(k) );
		path.push_back( temp.at(k+1) );
		path.push_back( temp.at(k+2) );
	}
	temp.clear();

	vector<int> temp2;
	for(unsigned int i = (path2D.size() - 1 ); i>=0; i--) {
		temp2.push_back(path2D.at(i));
		path2D.pop_back();
		if(i == 0) break;
	}

	for(unsigned int k = 0; k<temp2.size(); k++ ) {
		path2D.push_back( temp2.at(k) );	
	}
	temp2.clear();
  //================================
 // reverseVecOrder(path);
	open_nodes.clear();
	closed_nodes.clear();
	

 delete[] newNode;
 //delete[] currentNode;
 delete[] end;
 delete[] tile;
	
  return true;
}

//....................................................get real X co-ordinate.....................................................\\

double getRealDistX(int x, double Dist) {
		double pixelWidth = (Dist * 1.09723757325) / widthdepth; //2dtan(57.5/2) / 640
		return  x * pixelWidth;
}

//....................................................get real Y co-ordinate......................................................\\

double getRealDistY(int y, double Dist) {
		double pixelHeight = (Dist * 0.79791909194) / heightdepth; //2dtan(43.5/2) / 480
		return  y * pixelHeight;
}

//....................................................get pixel x-no. real X co-ordinate...........................................\\

int getXfromRealX(double x, double Dist) {
		double pixelWidth = (Dist * 1.08591139928) / widthdepth; //2dtan(57/2) / 640
		return (int) (x / pixelWidth);
}

//....................................................get Pixel y-no. from real Y co-ordinate......................................\\

int getYfromRealY(double y, double Dist) {
		double pixelHeight = (Dist * 0.78782095123) / heightdepth; //2dtan(43/2) / 480
		return (int) (y / pixelHeight);
}

//.......................................Show kinect Depth data on the screen.....................................................\\

static void drawKinectDatadepth() {

		NUI_IMAGE_FRAME depthFrame;
		NUI_LOCKED_RECT LockedRect;
		HRESULT hrD;
		hrD = sensor->NuiImageStreamGetNextFrame(depthStream, 0, &depthFrame);

		if (hrD != S_OK) exit(4);

		INuiFrameTexture* texture = depthFrame.pFrameTexture;
		texture->LockRect(0, &LockedRect, NULL, 0);
		
		if (LockedRect.Pitch != 0) {
			const USHORT* curr2 = (const USHORT*) LockedRect.pBits;
			fillbytebufferdepth(curr2);
		}
		

		//fillcharbufferdepth(dest);
		Mat xz(heightdepth, widthdepth,  CV_8UC4 , bufferdepth);
		Depth = xz;
		
		erode(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		erode(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		erode(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		erode(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		dilate(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		dilate(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		dilate(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		dilate(Depth,Depth,cv::Mat(), cv::Point(-1,-1));
		dilate(Depth,Depth,cv::Mat(), cv::Point(-1,-1));

		//Depth2 = cvCreateMat(heightdepth,widthdepth,CV_32FC1);
		//blur( Depth,Depth, Size(15,15) );
		//Mat diff = xy-Depth;
		 IplImage* frame2 = new IplImage(Depth);
		 cvSmooth(frame2, frame2, CV_GAUSSIAN,7,7);

		if(flagDepth) cvShowImage("Depth",frame2);
		
		//original = cvCreateImage(cvSize(widthdepth, heightdepth),IPL_DEPTH_8U, 4);
		original = new IplImage(Depth);
		//displayedImage = cvCreateImage(cvSize(widthdepth, heightdepth), IPL_DEPTH_8U, 4);
		cvSetZero( displayedImage );

		
		
		if(! BlobDetection(original)) exit(6);
		findSafeArea(nearestDst);
		//if(! trackColorObject()) exit(3);
		
		//if(! makeContoursDepth()) exit(2); 
		
	    //IplImage* frame3 = new IplImage(drawing2);
		//cvSmooth(frame3, frame3, CV_GAUSSIAN,3,3);
		//if(flagContoursDepth)  cvShowImage("Contours2",frame3);

		
		if(flagBlobs) cvShowImage("Blobs", displayedImage);
		cvAnd(displayedImage,accumelatedFrames , accumelatedFrames);
		
		xz.release();
		Depth.release();
		delete[] frame2;
		delete[] original;

		texture->UnlockRect(0);
		sensor->NuiImageStreamReleaseFrame(depthStream, &depthFrame);
		depth = !depth;
		
		//delete[] bufferdepth;
		//delete[] bufferdepth2;
}

//......................................Show kinect Color data on the screen......................................................\\

static void drawKinectDatacolor() {

		NUI_IMAGE_FRAME imageFrame;
		NUI_LOCKED_RECT LockedRect;
		HRESULT hrC = sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageFrame);
		
		if (hrC != S_OK) exit(5);

		INuiFrameTexture* texture = imageFrame.pFrameTexture;
		texture->LockRect(0, &LockedRect, NULL, 0);
		
		if (LockedRect.Pitch != 0) {
			const BYTE* curr = (const BYTE*) LockedRect.pBits;

			//fill buffer with curr
			fillBuffer(curr);
    }

	Mat	RGB_Contours(height1, width1,  CV_8UC4 , buffer);
	Mat imgRGB(height1, width1,  CV_8UC4 , buffer);
	cvtColor(imgRGB, imgRGB, CV_RGBA2RGB);
	cvtColor(imgRGB, imagegrey, CV_RGB2GRAY);

	
	Mat imageRGBmotion2(height1, width1, CV_8UC4 , buffer);
	imageRGBmotion = imageRGBmotion2;
	
	
	//RGBout = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
	//imgHSV = cvCreateImage(cvSize(width1, height1), IPL_DEPTH_8U, 3);
    //imgTracking = cvCreateImage(cvSize(width1, height1),IPL_DEPTH_8U, 3);
	//imageRGB = cvCreateImage(cvSize(width1, height1),IPL_DEPTH_8U, 3);
	imageRGB = new IplImage(imgRGB);
	motion = new IplImage(imageRGBmotion);
	//displayedImage = cvCreateImage(cvSize(widthdepth, heightdepth), IPL_DEPTH_8U, 4);
	img_curr = imageRGBmotion.clone();

	if (first) {
			img_prev = img_curr.clone();
			first = false;
   }
	
	//if(! CannyThreshold()) exit(1); 
	//if(! trackColorObject()) exit(3);
	//if(! makeContours()) exit(2); 
	//if(! BlobDetection(motion)) exit(4);
	 
	
	//IplImage* frame1 = new IplImage(edges1);
	IplImage* frame2 = new IplImage(imageRGBmotion);
	//IplImage* frame3 = new IplImage(drawing);
	
	
	//cvThreshold(frame2, frame2, 140, 255, CV_THRESH_BINARY);

	//cvSmooth(frame1, frame1, CV_GAUSSIAN,3,3);
	//cvSmooth(frame2, frame2, CV_GAUSSIAN,3,3);
	//cvSmooth(frame3, frame3, CV_GAUSSIAN,3,3);
	
	//display
	//if(flagEdge) cvShowImage("Edge", frame1);
	if(flagRGB) cvShowImage("RGB", frame2);
	//if(flagContours) cvShowImage("Contours",frame3);
	//if(flagcolorT) cvShowImage("ColorT", RGBout);
	//if(flagBlobs) cvShowImage("Blobs", displayedImage);

	waitKey(20);

	img_prev = img_curr.clone();
	
	cvReleaseImage(&imgHSV);
	cvReleaseImage(&imgTracking);
	imageRGBmotion.release();
	drawing.release();

	edges1.release();
	RGB_Contours.release();
	imageRGBmotion.release();
	edges2.release();
	imgRGB.release();
	imageRGBmotion.release();
	img_diff.release();
	imagegrey.release();
	img_curr.release();
	img_prev.release();
	
	delete[] frame2;
	delete[] imageRGB;
	delete[] motion;
	texture->UnlockRect(0);
	sensor->NuiImageStreamReleaseFrame(rgbStream, &imageFrame);
	depth = !depth;
	
}


void drawMap() {

	namedWindow("map",CV_WINDOW_AUTOSIZE);


	
	for(int r = 0; r<480; r++) {
		for( int c = 0; c<640; c++) {

			if( ((r >= 0 && r<=5) &&  (c>=0 && c<=285)) || ((r >= 0 && r<=5) &&  (c>=360 && c<640)) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			
			if( (r >= 0 && r <= 220) &&  ((c >= 0 && c<=5) || (c >= 634 && c<= 639) ) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			if( (r >= 0 && r <= 100) &&  ( (c >= 280 && c<=285) || (c >= 360 && c<=365) ) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			if( (r >= 140 && r < 220) &&  ( (c >= 280 && c<=285) || (c >= 360 && c<=365) ) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			  
			if( ((r >= 220 && r <= 225) &&  (c>=230 && c<=285)) || ((r >= 220 && r <= 225) &&  (c>=360 && c<=415)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			  if( (r > 220 && r < 260) &&  ((c >= 0 && c<=5) || (c >= 230 && c<=235) || (c >= 410 && c<=415) || (c >= 634 && c<= 639)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			  if( ((r >= 260 && r<= 265) &&  (c>=0 && c<=130)) || ((r >= 260 && r<= 265) &&  (c>=410 && c<=430)) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			  if( ((r >= 260 && r<= 265) &&  (c>=170 && c<=235)) || ((r >= 260 && r<= 265) &&  (c>=470 && c<=639)) ) cvSet2D(map,r,c,cvScalar(0,255,250));


			if( ((r >= 300 && r<= 305) &&  (c>=0 && c<=130)) || ((r >= 300 && r<= 305) &&  (c>=410 && c<=430)) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			if( ((r >= 300 && r<= 305) &&  (c>=170 && c<=235)) || ((r >= 300 && r<= 305) &&  (c>=470 && c<=639)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			if( (r > 300 && r <= 340) &&  ((c >= 0 && c<=5) || (c >= 230 && c<=235) || (c >= 410 && c<=415) || (c >= 634 && c<= 639)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			 if( ((r >= 340 && r <= 345) &&  (c>=230 && c<=285)) || ((r >= 340 && r <= 345) &&  (c>=360 && c<=415)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			if( (r > 340 && r <= 479) &&  ((c >= 0 && c<=5) || (c >= 634 && c<= 639) ) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			if( (r > 340 && r <= 400) &&  ( (c >= 280 && c<=285) || (c >= 360 && c<=365)  ) ) cvSet2D(map,r,c,cvScalar(0,255,250));
			if( (r > 440 && r <= 479) &&  ((c >= 280 && c<=285) || (c >= 360 && c<=365)  ) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			if( ((r >= 474 && r<=479) &&  (c>=0 && c<=285)) || ((r >= 474 && r<=479) &&  (c>=360 && c<640)) ) cvSet2D(map,r,c,cvScalar(0,255,250));

			//if( (r >= 100 && r <= 140) &&  ((c >= 280 && c<= 285) || (c >= 360 && c<= 365) ) ) cvSet2D(map,r,c,cvScalar(0,0,0));

			//if( ((r >= 260 && r<=265) &&  (c>=130 && c<=170)) || ((r >= 260 && r<=265) &&  (c>=430 && c<=470)) ) cvSet2D(map,r,c,cvScalar(0,0,0));

			//if( ((r >= 300 && r<=305) &&  (c>=130 && c<=170)) || ((r >= 300 && r<=305) &&  (c>=430 && c<=470)) ) cvSet2D(map,r,c,cvScalar(0,0,0));

			//if( (r >= 400 && r <= 440) &&  ((c >= 280 && c<= 285) || (c >= 360 && c<= 365) ) ) cvSet2D(map,r,c,cvScalar(0,0,0));

			if( (r >= 240 && r <= 320) && (c >= 260 && c <= 380) ) cvSet2D(map,r,c,cvScalar(0,255,250));
		
		}
	}

	cvShowImage("map", map);

}


void Astar() {

	 node* currentNode;
	 node* newNode;
	 node* end;
	 vector<node*> open_nodes;
	 vector<node*> closed_nodes;
	 double Gcost;
	 double Hcost;
	 double ParentGcost;
	 double tempGcost;
	 double minFcost;
	 int dx[8] = {5, 5, 0, -5, -5, -5, 0, 5};
	 int dy[8] = {0, -5, -5, -5, 0, 5, 5, 5};
	 int counter3 = 0;
	 int Index;
	 int index2;
	 int index4;

	 
	 end = new node(450, 410);

	 currentNode = new node(320, 10);
	 currentNode->setIndex(0);
	 currentNode->setParentIndex(-1);

	 ParentGcost = 0;

	 currentNode->setGcost(ParentGcost);
	 closed_nodes.push_back(currentNode);

	 
  bool flagDone = false;
  CvScalar scale;
  while(1) {

	  for(int k  = 0; k < 8; k++) {
		 
		   if ( (((currentNode->getX()) + dx[k]) < widthdepth) && (((currentNode->getX()) + dx[k]) >= 0) 
			&&    (((currentNode->getY()) + dy[k]) >= 0) && (((currentNode->getY()) + dy[k]) < heightdepth) ) {

		  newNode = new node(((currentNode->getX()) + dx[k]), ((currentNode->getY()) + dy[k]));
		  scale = cvGet2D(map, newNode->getY(), newNode->getX());
		  
		  if(scale.val[0] == 0 && scale.val[1] == 255 && scale.val[2] == 250 ) continue;

		  Hcost = sqrt( (double) ( (end->getX() - newNode->getX()) * (end->getX() - newNode->getX()) ) + (double) ( (end->getY() - newNode->getY())*(end->getY() - newNode->getY()) ) );
		  newNode->setHcost(Hcost);

		  if(newNode->getX() == end->getX()  && newNode->getY() == end->getY()) flagDone = true;

		   if( (! isElementInList(open_nodes, newNode))  &&  (! isElementInList(closed_nodes, newNode))  ) {
					 counter3++;
					 newNode->setIndex( counter3 );

					 Gcost = (currentNode->getGcost()) 
						 + sqrt( (double) ( (newNode->getX() - currentNode->getX()) * (newNode->getX() - currentNode->getX()) ) 
						 +  (double) ( (newNode->getY() - currentNode->getY()) * (newNode->getY() - currentNode->getY()) ) );

					
					 newNode->setGcost(Gcost);
					 newNode->setFcost();
					 newNode->setParentIndex(currentNode->getIndex());
					 open_nodes.push_back(newNode);  

					  
				}

			else if( isElementInList(open_nodes, newNode) ) {
					//check if beter to go direct from parent of parent
					
				tempGcost = (currentNode->getGcost()) 
						 + sqrt( (double) ( (newNode->getX() - currentNode->getX()) * (newNode->getX() - currentNode->getX()) ) 
						 +  (double) ( (newNode->getY() - currentNode->getY()) * (newNode->getY() - currentNode->getY()) ) );


				Index = findElementIndexInList(open_nodes, newNode);
				if(Index == -1)  exit(881);
					
				if(tempGcost < (open_nodes.at(Index)->getGcost()) ) {
					  
					  open_nodes.at(Index)->setGcost(tempGcost);
					  open_nodes.at(Index)->setFcost();
					  open_nodes.at(Index)->setParentIndex(currentNode->getIndex());
					
				 }
			  }

			if(flagDone  == true){
					closed_nodes.push_back(newNode);
					break;
				}

	      }
	  }


	  if(flagDone  == true) {
			break;
		}


		minFcost = (open_nodes.at(0))->getFcost();
		index2 = 0;
		for(unsigned int j = 0; j<open_nodes.size(); j++) {
			if( (open_nodes.at( j))->getFcost() < minFcost ) {
				minFcost = (open_nodes.at(j))->getFcost();
				index2 = j;
			}
		  }

				
		
		currentNode = new node();
		currentNode->setX( (open_nodes.at(index2))->getX() );
		currentNode->setY( (open_nodes.at( index2))->getY() );
		currentNode->setIndex( (open_nodes.at( index2))->getIndex() );
		currentNode->setParentIndex( (open_nodes.at( index2))->getParentIndex() );
		currentNode->setGcost( (open_nodes.at(index2))->getGcost() );
		currentNode->setHcost( (open_nodes.at( index2))->getHcost() );
		currentNode->setFcost( );

	
		//check if it was easier to go directly from start
		
			//add to closed
			closed_nodes.push_back(currentNode);

			//remove from open list
			
		
			index4 = findElementIndexInList(open_nodes, currentNode);
			if(index4 == -1)  exit(55);
			open_nodes.erase(open_nodes.begin() + index4);


			if(open_nodes.size() == 1){
				break;
			}

     }

  AstarPath.clear();

  
  AstarPath.push_back(end->getY());
  AstarPath.push_back(end->getX());
 
 node* tile;
 

 tile  = closed_nodes.at(closed_nodes.size() - 1);

  int parentIndex;
  unsigned int j;
  
  while(1) {
    
	 AstarPath.push_back(tile->getY());
	 AstarPath.push_back(tile->getX());
	 
	 if(tile->getParentIndex() == -1) break;

	 parentIndex = tile->getParentIndex();
	 j = 0;
	 for(j = 0; j<closed_nodes.size(); j++) {
			if( (closed_nodes.at(j))->getIndex() ==  parentIndex ) {
				break;
			}
	 }
	 
	tile = closed_nodes.at(j);
	
  }


  vector<int> temp2;
	for(unsigned int i = (AstarPath.size() - 1 ); i>=0; i--) {
		temp2.push_back(AstarPath.at(i));
		AstarPath.pop_back();
		if(i == 0) break;
	}

	for(unsigned int k = 0; k<temp2.size(); k++ ) {
		AstarPath.push_back( temp2.at(k) );	
	}
	temp2.clear();
	open_nodes.clear();
	closed_nodes.clear();
	
	//AstarextendPath();

	for(unsigned int k = 0; k<=AstarPath.size()-4; k = k + 2 ) {
		cvLine(map, cv::Point (AstarPath.at(k), AstarPath.at(k+1)), cv::Point (AstarPath.at(k+2), AstarPath.at(k+3)), cvScalar(255, 0, 0), 5, 8, 0);	
	}


	// convert to 3D
	AstarPath3D.clear();
	for(unsigned int j = 0; j < AstarPath.size()-1; j = j+2) {

		//(x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
		AstarPath3D.push_back( ((AstarPath.at(j) - 0) * (50000 - (0)) / (widthdepth - 0)) + (0) ); // x
		AstarPath3D.push_back( ((AstarPath.at(j+1) - 0) * (50000 - (0)) / (heightdepth - 0)) + (0) ); // z
	}

 delete[] newNode;
 //delete[] currentNode;
 delete[] end;
 delete[] tile;
	
}


void localize(double duration, double accZ, double uZ,  double accX, double uX, double intialX, double intialZ) {
	//localizizng by calculating velocity assuming const acceleration
	/*time_t t = time(0);///3600000;
	struct tm * timeinfo = localtime ( &t );*/
	//asctime (timeinfo);
	
	//std::clock_t start;
 //   double duration;
 //   start = std::clock();


	// duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	double Vx = uX + accX * duration;
	//double Vy = uY + accY * duration;
	double Vz = uZ + accZ * duration;
	
	currVelocities.clear();
	//currLocation.resize(2);
	currVelocities.push_back(Vx);
	//currLocation.push_back(currY);
	currVelocities.push_back(Vz);


	double currZ = (((Vz * Vz) - (uZ * uZ)) / (2 * accZ)) + intialZ;
	double currX = (((Vx * Vx) - (uX * uX)) / (2 * accX)) + intialX;
	//double currY = (((Vy * Vy) - (uY * uY)) / (2 * accY)) + intialY;

	currLocation.clear();
	//currLocation.resize(2);
	currLocation.push_back(currX);
	//currLocation.push_back(currY);
	currLocation.push_back(currZ);
	
}

void localize2(double duration, double intialX, double intialZ, double accX, double accZ) {
	
	double currZ;
    double currX;
	
	int counter2 = 0;
	for(unsigned int i = 0; i<=AstarPath3D.size()-2; i = i+2) {
		if(/*AstarPath3D.at(i) >= intialX ||*/ AstarPath3D.at(i+1) >= intialZ) {
			counter2 = i;
			break;
		}
		
	}
	if (counter2 == 0){
		
		counter2 = 2;

	}
		currZ = ( (Vvector.at(counter2+1) * duration)  + (0.5 * accZ * (duration *duration)) ) + intialZ;
        currX = ( (Vvector.at(counter2) * duration)  + (0.5 * accX * (duration *duration)) ) + intialX;


	currLocation.clear();
	//currLocation.resize(2);
	currLocation.push_back( currX );
	//currLocation.push_back(currY);
	currLocation.push_back( currZ );

}

void setVelocityVector() {

	Vvector.push_back(0);
	Vvector.push_back(0);

	for(unsigned int i = 0; i <= AstarPath.size()-4; i = i + 2 ) {

		if( (AstarPath.at(i) == AstarPath.at(i+2)) && AstarPath.at(i+1) < AstarPath.at(i+3) ) {
			
			Vvector.push_back(0);
			Vvector.push_back(1500);

		}

		else if( (AstarPath.at(i) == AstarPath.at(i+2)) && AstarPath.at(i+1) > AstarPath.at(i+3) ) {
			
			Vvector.push_back(0);
		    Vvector.push_back(-1500);

		}

		else if( (AstarPath.at(i) > AstarPath.at(i+2)) && AstarPath.at(i+1) < AstarPath.at(i+3) ) {
			
			Vvector.push_back(-1500);
		    Vvector.push_back(1500);

		}

		else if( (AstarPath.at(i) > AstarPath.at(i+2)) && AstarPath.at(i+1) > AstarPath.at(i+3) ) {
			
			Vvector.push_back(-1500);
		    Vvector.push_back(-1500);

		}

		else if( (AstarPath.at(i) < AstarPath.at(i+2)) && AstarPath.at(i+1) < AstarPath.at(i+3) ) {
			
			Vvector.push_back(1500);
		    Vvector.push_back(1500);

		}

		else if( (AstarPath.at(i) < AstarPath.at(i+2)) && AstarPath.at(i+1) > AstarPath.at(i+3) ) {
			
			Vvector.push_back(1500);
		    Vvector.push_back(-1500);

		}
		else if( (AstarPath.at(i) > AstarPath.at(i+2)) && AstarPath.at(i+1) == AstarPath.at(i+3) ) {
			
			Vvector.push_back(-1500);
		    Vvector.push_back(0);

		}

	    else if( (AstarPath.at(i) < AstarPath.at(i+2)) && AstarPath.at(i+1) == AstarPath.at(i+3) ) {
			
			Vvector.push_back(1500);
		    Vvector.push_back(0);

		}

	 else {

			Vvector.push_back(0);
		    Vvector.push_back(0);

	   }

  }

	//Vvector.push_back(0);
	//Vvector.push_back(0);

}


//================================================================================================\\
//======================================= M A I N ================================================\\
//================================================================================================\\

int main(int argc, char* argv[]) {
	double intialX;
	double intialZ;
	double uX;
	double uZ;
	double accZ;
	double accX;
	int intialX2;
	int intialZ2;

	int count = 1;

	drawMap();
	Astar();
	setVelocityVector();
	drawMap();

	intialX = AstarPath3D.at(0);
	intialZ = AstarPath3D.at(1);

	uX = 0;
	uZ = 0;

	accZ = 0;
	accX = 0;


	//localize();
	if (!initKinect()) return 1;
	//namedWindow("Edge",CV_WINDOW_AUTOSIZE);
	namedWindow("RGB",CV_WINDOW_AUTOSIZE);
	namedWindow("Depth",CV_WINDOW_AUTOSIZE);
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//namedWindow("Contours2", CV_WINDOW_AUTOSIZE);
	//namedWindow("ColorT", CV_WINDOW_AUTOSIZE);
	//namedWindow("CP",1);
	//namedWindow("accumelated",CV_WINDOW_AUTOSIZE);
	namedWindow("Blobs",CV_WINDOW_AUTOSIZE);
	
	lowThreshold = 300;

	//cv::moveWindow("Contours",  660,  0);
	cv::moveWindow("RGB",  0,  520);
	//cv::moveWindow("Edge",  660,  520);
	//cv::moveWindow("colorT",  0,  0);
	cv::moveWindow("Depth",  0,  0);
	//cv::moveWindow("CP",  1320,  300);
	cv::moveWindow("Blobs",  1000, 0);
	
	/*glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
	glutInitWindowSize(640,480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);*/


	glutInit(&argc, argv);        // Initialize glut
            // Setup display mode to double buffer and RGB color
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);  
            // Set the screen size
      glutInitWindowSize(600, 600);                                        
      glutCreateWindow("3D Path");

	//cvNamedWindow( "FaceDet", CV_WINDOW_AUTOSIZE);

	//HWND hwnd1 = (HWND)cvGetWindowHandle("Edge");
	HWND hwnd2 = (HWND)cvGetWindowHandle("RGB");
	//HWND hwnd3 = (HWND)cvGetWindowHandle("Contours");
	//HWND hwnd4 = (HWND)cvGetWindowHandle("ColorT");
	HWND hwnd5 = (HWND)cvGetWindowHandle("Depth");
	HWND hwnd6 = (HWND)cvGetWindowHandle("Blobs");
	//HWND hwnd7 = (HWND)cvGetWindowHandle("Contours2");
	
	//createTrackbar( "Threshold", "CP", &lowThreshold,500 , NULL);
	//cvCreateTrackbar("LowerH", "CP", &lowerH, 180, NULL);
	//cvCreateTrackbar("UpperH", "CP", &upperH, 180, NULL);
	//cvCreateTrackbar("LowerS", "CP", &lowerS, 255, NULL);
	//cvCreateTrackbar("UpperS", "CP", &upperS, 255, NULL);
	//cvCreateTrackbar("LowerV", "CP", &lowerV, 255, NULL);
	//cvCreateTrackbar("UpperV", "CP", &upperV, 255, NULL); 

	
    // Main loop
	
	cvSetZero( accumelatedFrames );
	cvNot( accumelatedFrames, accumelatedFrames );


	cvLine(map, cv::Point (((intialX - 0) * (widthdepth - 0) / (50000 - 0)), ((intialZ - 0) * (heightdepth - 0) / (50000 - 0))), 
		cv::Point (((intialX - 0) * (widthdepth - 0) / (50000 - 0)), ((intialZ - 0) * (heightdepth - 0) / (50000 - 0))), cvScalar(0, 0, 255), 10, 8, 0);

	while (flagRGB || flagDepth) {
		  
	

	/*  if(counter == 10) {
		  cvShowImage("accumelated",accumelatedFrames);
		  cvSetZero( accumelatedFrames );
		  cvNot( accumelatedFrames, accumelatedFrames );
		  counter = 0;
	   }*/
		//initializeSensor(depth);
		if(depth){
			std::clock_t start;
			double duration;
			start = std::clock();

			//waitKey(500);
			duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
					//waitKey(200);
			WaitForSingleObject(depthEvent, INFINITE);
			
			localize2(0.2, intialX, intialZ, accX, accZ);
			drawKinectDatadepth();
			
	     	    intialX = currLocation.at(0);
			    intialZ = currLocation.at(1);
				intialX2 = (int)intialX -  newExtendedPath3D.at(newExtendedPath3D.size()-3);
				intialZ2 = (int)intialZ + newExtendedPath3D.at(newExtendedPath3D.size()-1);
				intialX2 =  (intialX - 0) * (widthdepth - 0) / (50000 - 0);
				intialZ2 =  (intialZ - 0) * (heightdepth - 0) / (50000 - 0);

				newExtendedPath3D.clear();
			
			cvLine(map, cv::Point (intialX2, intialZ2), cv::Point (intialX2, intialZ2), cvScalar(0, 0, 255), 10, 8, 0);
			//maps/
			drawMap();
			//drawKinectDatadepth();
			
			//=========================
			//== kalman filter by doula =//

				predX = A * updatedX + B * U;
				predCovX = A * updatedCovX + R;

				predY = A * updatedY + B * U;
				predCovY = A * updatedCovY + R;

				Kx = (predCovX * C) / (C * predCovX + Q);
				Ky = (predCovY * C) / (C * predCovY + Q);

				updatedX = predX + Kx * ((kalmanInputX) - (C * predX));
				updatedY = predY + Ky * ((kalmanInputY) - (C * predY));
				
				updatedCovX = (1 - Kx * C) * predCovX;
				updatedCovY = (1 - Ky * C) * predCovY;

				//==//

			//=========================
			//===========================

			//we initizlilze the glut. functions
			//glutInit(&argc, argv);
			//glutS\\etOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
			//glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
		/*	glutInitWindowSize(640,480);
			glutInitWindowPosition(100, 100);
			glutCreateWindow(argv[0]);
		*/

			/*init();
			glutDisplayFunc(DrawCube);
			glutReshapeFunc(reshape);
				*/
 
			init();
            glutReshapeFunc(reshape);
            glutDisplayFunc(drawings);
            // Set window's key callback
            glutKeyboardFunc(keyboard);  
            // Set window's to specialKey callback   
            glutSpecialFunc(specialKey);  
             // Set window's to Mouse callback
           //glutMouseFunc(new MouseCallback(processMouseActiveMotion));   
            // Set window's to motion callback
            //glutMotionFunc(processMouse);             
            // Set window's to mouse motion callback
            glutMouseWheelFunc(processMouseWheel);
           // glutMainLoop();

			//Set the function for the animation.
			//glutIdleFunc(animation);
			glutMainLoopEvent();
			 //glutMainLoop();
			//glutDestroyWindow(glutGetWindow());
			//===========================

			if (/* (abs( (currLocation.at(0) - AstarPath3D.at(AstarPath3D.size()-2)) ) <= 2000) &&*/ (currLocation.at(1) >= AstarPath3D.at(AstarPath3D.size()-1)) ) break;
		}
	      else {
			WaitForSingleObject(rgbEvent, INFINITE);
			drawKinectDatacolor();
			
		}
		
		//if(!IsWindowVisible(hwnd1)) flagEdge = false;
		if(!IsWindowVisible(hwnd2)) flagRGB = false;
		//if(!IsWindowVisible(hwnd3)) flagContours = false;
		//if(!IsWindowVisible(hwnd4)) flagcolorT = false;
		if(!IsWindowVisible(hwnd5)) flagDepth = false;
		if(!IsWindowVisible(hwnd6)) flagBlobs = false;
		//if(!IsWindowVisible(hwnd7)) flagContoursDepth = false;

	//delete[] timeinfo;
	}

	cvDestroyAllWindows();
	sensor->NuiShutdown();

	/*delete[] motion;
	delete[] imgTracking;
	delete[] RGBout;
	delete[] imageRGB;
	delete[] imgHSV;
	delete[] imgThresh;
	delete[] accumelatedFrames;
	delete[] originalThr;*/
		 
	return 0;
}