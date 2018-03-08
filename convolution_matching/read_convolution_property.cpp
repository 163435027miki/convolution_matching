#include <stdio.h>
#include<fstream>
#include<iostream>
#include<string>
#include<sstream>	//文字ストリーム
#include <direct.h>//フォルダを作成する
#include <tuple>

using namespace std;
#include <opencv2/opencv.hpp>	//画像読み込み


std::tuple<char&,int, int,int,int> read_log(int paramerter[], int paramerter_count, int sd, char date[], char Inputimage[]) {

	printf("畳み込みを行わず，データを読み取ります\n");
	int image_x, image_y, image_xt, image_yt;
	//char Inputimage[128]="";
	char no_date[255];
/*
	//結果を保存するフォルダの作成
	//フォルダ名は実行日時になる
	sprintf(date_directory, "..\\result_usa\\%s\\", date);
	if (_mkdir(date_directory) == 0) {
		printf("フォルダ %s を作成しました\n", date_directory);
	}
	else {
		printf("フォルダ作成に失敗しました。もしくは作成済みです\n");
	}
	*/
	//char conv_dire1[128];
	char conv_dire0[255];
	char conv_dire1[255];
	char conv_image_dire[255];
	char conv_template_dire[255];
	if(paramerter[0]==0)sprintf(conv_dire0, "..\\result_usa\\%s\\%dk_conv_sd%d\\", date,paramerter[paramerter_count],sd);
	if (paramerter[0] == 1 ||  paramerter[0] == 2)sprintf(conv_dire0, "..\\result_usa\\%s\\%d×%dsobel_conv_sd%d\\", date, paramerter[paramerter_count], paramerter[paramerter_count], sd);
	
	sprintf(conv_dire1, "%slog.txt", conv_dire0);
	sprintf(conv_image_dire, "%sV(0).bmp", conv_dire0);
	sprintf(conv_template_dire, "%sV(0)t.bmp", conv_dire0);


	std::ifstream conv_dire2;
	conv_dire2.open(conv_dire1, ios::in);

	char conv_dire3[255];

	char Inputtemplate[128];
	int count_convolution_log = 0;

	//プロパティtxtファイルの読み込み
	if (conv_dire2.fail())
	{
		printf("propertyテキストを読み取ることができません\n");
		printf("propertyテキスト : %s\n", conv_dire1);
	}
	while (conv_dire2.getline(conv_dire3, 256 - 1)) {


		
		if (count_convolution_log == 3)sprintf(Inputimage, conv_dire3);	//使用した探索対象画像名
		if (count_convolution_log == 6)sprintf(Inputtemplate, conv_dire3);	//使用したテンプレート画像
		

		++count_convolution_log;
	}

	
	conv_dire2.close();

	//入力画像を読み込み
	cv::Mat ImputImageM = cv::imread(conv_image_dire);	//入力画像の読み込み．opencv
	image_x = ImputImageM.cols;
	image_y = ImputImageM.rows;

	//テンプレート画像を読み込み
	cv::Mat ImputImageT = cv::imread(conv_template_dire);	//入力画像の読み込み．opencv
	image_xt = ImputImageT.cols;
	image_yt = ImputImageT.rows;

	printf("convolution_log=%s\n", conv_dire1);
	printf("inputimage=%s\n", Inputimage);
	printf("image_x=%d,image_y=%d\nimage_xt=%d,image_yt=%d\n", image_x, image_y, image_xt, image_yt);

	return std::forward_as_tuple(*Inputimage, image_x, image_y, image_xt, image_yt);

}