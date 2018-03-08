#include <stdio.h>
#include<fstream>
#include<iostream>
#include<string>
#include<sstream>	//�����X�g���[��
#include <direct.h>//�t�H���_���쐬����
#include <tuple>

using namespace std;
#include <opencv2/opencv.hpp>	//�摜�ǂݍ���


std::tuple<char&,int, int,int,int> read_log(int paramerter[], int paramerter_count, int sd, char date[], char Inputimage[]) {

	printf("��ݍ��݂��s�킸�C�f�[�^��ǂݎ��܂�\n");
	int image_x, image_y, image_xt, image_yt;
	//char Inputimage[128]="";
	char no_date[255];
/*
	//���ʂ�ۑ�����t�H���_�̍쐬
	//�t�H���_���͎��s�����ɂȂ�
	sprintf(date_directory, "..\\result_usa\\%s\\", date);
	if (_mkdir(date_directory) == 0) {
		printf("�t�H���_ %s ���쐬���܂���\n", date_directory);
	}
	else {
		printf("�t�H���_�쐬�Ɏ��s���܂����B�������͍쐬�ς݂ł�\n");
	}
	*/
	//char conv_dire1[128];
	char conv_dire0[255];
	char conv_dire1[255];
	char conv_image_dire[255];
	char conv_template_dire[255];
	if(paramerter[0]==0)sprintf(conv_dire0, "..\\result_usa\\%s\\%dk_conv_sd%d\\", date,paramerter[paramerter_count],sd);
	if (paramerter[0] == 1 ||  paramerter[0] == 2)sprintf(conv_dire0, "..\\result_usa\\%s\\%d�~%dsobel_conv_sd%d\\", date, paramerter[paramerter_count], paramerter[paramerter_count], sd);
	
	sprintf(conv_dire1, "%slog.txt", conv_dire0);
	sprintf(conv_image_dire, "%sV(0).bmp", conv_dire0);
	sprintf(conv_template_dire, "%sV(0)t.bmp", conv_dire0);


	std::ifstream conv_dire2;
	conv_dire2.open(conv_dire1, ios::in);

	char conv_dire3[255];

	char Inputtemplate[128];
	int count_convolution_log = 0;

	//�v���p�e�Btxt�t�@�C���̓ǂݍ���
	if (conv_dire2.fail())
	{
		printf("property�e�L�X�g��ǂݎ�邱�Ƃ��ł��܂���\n");
		printf("property�e�L�X�g : %s\n", conv_dire1);
	}
	while (conv_dire2.getline(conv_dire3, 256 - 1)) {


		
		if (count_convolution_log == 3)sprintf(Inputimage, conv_dire3);	//�g�p�����T���Ώۉ摜��
		if (count_convolution_log == 6)sprintf(Inputtemplate, conv_dire3);	//�g�p�����e���v���[�g�摜
		

		++count_convolution_log;
	}

	
	conv_dire2.close();

	//���͉摜��ǂݍ���
	cv::Mat ImputImageM = cv::imread(conv_image_dire);	//���͉摜�̓ǂݍ��݁Dopencv
	image_x = ImputImageM.cols;
	image_y = ImputImageM.rows;

	//�e���v���[�g�摜��ǂݍ���
	cv::Mat ImputImageT = cv::imread(conv_template_dire);	//���͉摜�̓ǂݍ��݁Dopencv
	image_xt = ImputImageT.cols;
	image_yt = ImputImageT.rows;

	printf("convolution_log=%s\n", conv_dire1);
	printf("inputimage=%s\n", Inputimage);
	printf("image_x=%d,image_y=%d\nimage_xt=%d,image_yt=%d\n", image_x, image_y, image_xt, image_yt);

	return std::forward_as_tuple(*Inputimage, image_x, image_y, image_xt, image_yt);

}