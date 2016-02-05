#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
# include <cstdlib>
# include <iomanip>
# include <fstream>
# include <cmath>
# include <ctime>
# include <cstring>
#include <sstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>
#include <thread>

using namespace std;
using namespace cv;
using namespace Eigen;

int main(int argc, char** argv)
{
	ifstream myfile;
	myfile.open("mnist_train.csv");
	int Number = 1;
	Mat image = Mat(28,28, CV_8U, int(1));
	string val;

	ofstream myfile1;
	myfile1.open("Train.txt");
	ofstream myfile2;
	myfile2.open("Test.txt");
	
	while(myfile >> val)
	{
		int r = 0;
		int c = 0;
		string sub = "";
		string label = val.substr(0,1);
		for(int i = 2; i < val.length(); i++)
		{
		
			if(val[i] != ',')
				sub += val[i];
			else
			{
				int Result;	
				istringstream convert(sub);
				if ( !(convert >> Result) )	
					Result = 0;
				image.at<uchar>(r,c) = Result;
			
				c++;
				if(c == 28)
				{
					r++;
					c = 0;
				} 	
				sub = "";
			}

		
		}
		string Result;
		ostringstream convert;
		convert << Number;
		Result = convert.str();
		
		string dataset;
		if(Number <= 55000)
		{
			dataset = "Dataset/Train/" + Result + ".png";
			myfile1 << dataset << " " << label << endl;
		}
		else
		{
			dataset = "Dataset/Test/" + Result + ".png";
			myfile2 << dataset << " " << label << endl;
		}
		cout << dataset << endl;
		imwrite(dataset,image);
		Number++;
	}	
	
	waitKey(0);
	return 0;
}
