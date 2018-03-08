#include <string>
#include<sstream> //�����X�g���[��
#include<fstream>
#include<iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <stdio.h>

//�������m�ۂ��s�����߂̃w�b�_
#define ANSI				
#include "nrutil.h"	


using namespace std;

//opencv�̂��ߕK�v
int argc;
char** argv;

	

int cols;
int rows;
int make_image_offset;
int image_number;
//int make_image_repeat=3;


char Output_Image_name[128];
char Input_Image_name[128];						//���̓t�@�C�����E���͌��̐ݒ�
char input_csv_name[128];
char iutput_csv_name[128];
//static int makeimage[1024];
char use_name[128];

int ikkaioki = 0;
int ikkaioki_2 = 0;



void make_bmp(char date_directory2[],char Filename1[],int &image_x,int &image_y,int make_image_repeat,int Save_image_flag[][9]) {

	

	++ikkaioki;
	
	if(Save_image_flag[2][make_image_repeat]==0)return;		//����̉ӏ��̉摜���쐬���Ȃ����ɋA��
	
////////////////////////////////�����ݒ�//////////////////////////////////////////////////////////////////
	cols=image_x;			//����
	rows=image_y;			//�c��
	make_image_offset=128;	//�������Ȃ������ꏊ(0)�̉�f�l

	//cols=320;				//����
	//rows=128;				//�c��

	//�ǂݎ��csv�t�@�C���̖��O���w��
	image_number=45*(make_image_repeat-1);
	sprintf(use_name,"V(%d)",image_number);
	if (ikkaioki == 2) {
		sprintf(use_name, "V(%d)t", image_number);
		ikkaioki = 0;
	}

	std::vector<double>makeimage;
	makeimage.resize(cols);

	
//////////////////���o�̓t�@�C���̃f�B���N�g������//////////////////////////////////////////////////////////

	sprintf(Output_Image_name,"%s%s.bmp",date_directory2,use_name);	//�o�͉摜.bmp
	
	sprintf(iutput_csv_name,"%s%s.csv",date_directory2,use_name);		//���͉摜�̉�f�l.csv

	

////////////////////////InputImage�̉�f�l�̎擾////////////////////////////////////////////////////////////////////////////////////
	
	//static int OutputRGB[1024][1024];

	double **OutputRGB = matrix(0, image_x - 1, 0, image_y - 1);

	for (int j = 0; j < image_y; ++j) {
		for (int i = 0; i < image_x; ++i) {
			OutputRGB[i][j] = 0;
		}
	}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////


	cv::Mat make_image = cv::Mat::zeros(rows, cols, CV_8UC1);


	int i=0,j=0;

	int count_small=0;
	ifstream make(iutput_csv_name);
	if(!make){printf("%s������܂���\n",iutput_csv_name);exit(1);}

	string str_make;
		while(getline(make,str_make)){
		string token_make;
        istringstream stream_make(str_make);
		
        while(getline(stream_make,token_make,',')){	//��s�ǂݎ��Dmakeimage�̂݁C�J��Ԃ��͈͎̔w��ɗp����
			int tmp_make=stof(token_make);			//�����𐔎��ɕϊ�
			makeimage[count_small]=tmp_make;				//�z��ɑ��
			
			OutputRGB[count_small][j]=makeimage[count_small]+make_image_offset;
			
			if(OutputRGB[count_small][j]>255)OutputRGB[count_small][j]=255;
			if(OutputRGB[count_small][j]<0)OutputRGB[count_small][j]=0;

			

			make_image.at<uchar>(j, i) = OutputRGB[count_small][j];

			++count_small;
			++i;
		}++j;count_small=0;i=0;
		}				
   


		FILE *fp_Output_image;
		fp_Output_image = fopen( Output_Image_name, "w" );
	if( fp_Output_image == NULL ){
		printf( "%s�t�@�C�����J���܂���\n", Output_Image_name );
		//exit(1);
	}
	for(int y=0;y<rows;y++){
		for(int x=0;x<cols;x++){

			fprintf( fp_Output_image, "%d,",OutputRGB[x][y]);
		}
			fprintf( fp_Output_image, "\n");
	}

	fclose( fp_Output_image );

	printf( "%s�t�@�C���������݂��I���܂���\n", Output_Image_name );

	imwrite(Output_Image_name,make_image); 

	 //���O������
	//std::string windowName = "make_image";
	//std::string windowName = use_name;
	
	//cv::namedWindow(windowName);

	//��ʕ\��
	//cv::imshow(windowName,make_image);
	 
 
	
		//cv::waitKey(0);

	free_matrix(OutputRGB, 0, image_x - 1, 0, image_y - 1);
	
	
}

void chika_bmp(char date_directory2[], char date_directory3[], int &image_x, int &image_y, int make_image_repeat, double **V_2chika) {
	++ikkaioki_2;

	

	////////////////////////////////�����ݒ�//////////////////////////////////////////////////////////////////
	cols = image_x;			//����
	rows = image_y;			//�c��
	make_image_offset = 0;	//�������Ȃ������ꏊ(0)�̉�f�l

								//cols=320;				//����
								//rows=128;				//�c��

								//�ǂݎ��csv�t�@�C���̖��O���w��
	image_number = 45 * (make_image_repeat - 1);
	sprintf(use_name, "V(%d)", image_number);
	if (ikkaioki_2 == 2) {
		sprintf(use_name, "V(%d)t", image_number);
		ikkaioki_2 = 0;
	}



	//////////////////���o�̓t�@�C���̃f�B���N�g������//////////////////////////////////////////////////////////

	sprintf(Output_Image_name, "%s%s_2.bmp", date_directory3, use_name);	//�o�͉摜.bmp

	sprintf(iutput_csv_name, "%s%s.csv", date_directory2, use_name);		//���͉摜�̉�f�l.csv



																			////////////////////////InputImage�̉�f�l�̎擾////////////////////////////////////////////////////////////////////////////////////

	//static int OutputRGB[1024][1024];

	double **OutputRGB = matrix(0, cols - 1, 0, rows - 1);

	for (int j = 0; j < rows; ++j) {
		for (int i = 0; i < cols; ++i) {
			OutputRGB[i][j] = 0;
		}
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////


	cv::Mat make_image = cv::Mat::zeros(rows, cols, CV_8UC1);


	int i = 0, j = 0;

	int count_small = 0;
	ifstream make(iutput_csv_name);
	if (!make) { printf("%s������܂���\n", iutput_csv_name); exit(1); }

	for (int j = 0; j < image_y; ++j) {
		for (int i = 0; i < image_x; ++i) {
			OutputRGB[i][j] = V_2chika[i][j] + make_image_offset;

			if (OutputRGB[i][j]>255)OutputRGB[i][j] = 255;
			if (OutputRGB[i][j]<0)OutputRGB[i][j] = 0;

			make_image.at<uchar>(j, i) = OutputRGB[i][j];
		}
	}

	/*
	string str_make;
	while (getline(make, str_make)) {
		string token_make;
		istringstream stream_make(str_make);

		while (getline(stream_make, token_make, ',')) {	//��s�ǂݎ��Dmakeimage�̂݁C�J��Ԃ��͈͎̔w��ɗp����
			int tmp_make = stof(token_make);			//�����𐔎��ɕϊ�
		//	makeimage[count_small] = tmp_make;				//�z��ɑ��

			//OutputRGB[count_small][j] = makeimage[count_small] + make_image_offset;
			OutputRGB[count_small][j] = V_2chika[count_small][j] + make_image_offset;

			if (OutputRGB[count_small][j]>255)OutputRGB[count_small][j] = 255;
			if (OutputRGB[count_small][j]<0)OutputRGB[count_small][j] = 0;



			make_image.at<uchar>(j, i) = OutputRGB[count_small][j];

			++count_small;
			++i;
		}++j; count_small = 0; i = 0;
	}

	*/

	FILE *fp_Output_image;
	fp_Output_image = fopen(Output_Image_name, "w");
	if (fp_Output_image == NULL) {
		printf("%s�t�@�C�����J���܂���\n", Output_Image_name);
		//exit(1);
	}
	for (int y = 0; y<rows; y++) {
		for (int x = 0; x<cols; x++) {

			fprintf(fp_Output_image, "%d,", OutputRGB[x][y]);
		}
		fprintf(fp_Output_image, "\n");
	}

	fclose(fp_Output_image);

	printf("%s�t�@�C���������݂��I���܂���\n", Output_Image_name);

	imwrite(Output_Image_name, make_image);


	free_matrix(OutputRGB, 0, cols - 1, 0, rows - 1);
	//���O������
	//std::string windowName = "make_image";
	//std::string windowName = use_name;

	//cv::namedWindow(windowName);

	//��ʕ\��
	//cv::imshow(windowName,make_image);



	//cv::waitKey(0);




}