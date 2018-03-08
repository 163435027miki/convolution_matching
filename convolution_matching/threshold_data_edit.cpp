
//�������m�ۂ��s�����߂̃w�b�_
#define ANSI				
#include "nrutil.h"	

#include<stdio.h>
#include<math.h>
#include <omp.h>	//openMP

#include<fstream>
#include<iostream>
#include<string>
#include<sstream> //�����X�g���[��

#include <tuple>

#include<time.h>//���Ԃ�p����

#include <direct.h>//�t�H���_���쐬����

#include<vector>
using namespace std;


int threshold_data_edit(int image_xt, int image_yt, double **threshold_edit, double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t,int use_convolution_direction_flag[]) {

	printf("****************************************\n");
	printf("start�F threshold����p�f�[�^�̍쐬\n");
	printf("****************************************\n");

	//臒l����ɗp����֐�
	
	//������
	for (int i = 0; i < image_yt; i++) {
		for (int j = 0; j < image_xt; j++) {
			threshold_edit[j][i] = 0;
		}
	}
	
	switch (use_convolution_direction_flag[0]) {
	//��������݂̂��G�b�W���x�ɗp����
	case 1:
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {
				if (use_convolution_direction_flag[1] == 1)threshold_edit[j][i] += pow(V0t[j][i], 2);
				if (use_convolution_direction_flag[2] == 1)threshold_edit[j][i] += pow(V45t[j][i], 2);
				if (use_convolution_direction_flag[3] == 1)threshold_edit[j][i] += pow(V90t[j][i], 2);
				if (use_convolution_direction_flag[4] == 1)threshold_edit[j][i] += pow(V135t[j][i], 2);
				if (use_convolution_direction_flag[5] == 1)threshold_edit[j][i] += pow(V180t[j][i], 2);
				if (use_convolution_direction_flag[6] == 1)threshold_edit[j][i] += pow(V225t[j][i], 2);
				if (use_convolution_direction_flag[7] == 1)threshold_edit[j][i] += pow(V270t[j][i], 2);
				if (use_convolution_direction_flag[8] == 1)threshold_edit[j][i] += pow(V315t[j][i], 2);
				threshold_edit[j][i] = sqrt(threshold_edit[j][i]);
			}
		}
	//�S�������G�b�W���x�Ƃ��ėp����
	case 0:
	default:
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {
				threshold_edit[j][i] = sqrt(pow(V0t[j][i], 2) + pow(V45t[j][i], 2) + pow(V90t[j][i], 2) + pow(V135t[j][i], 2) + pow(V180t[j][i], 2) + pow(V225t[j][i], 2) + pow(V270t[j][i], 2) + pow(V315t[j][i], 2));
			}
		}
	}

	return **threshold_edit;

	

}

std::tuple<int, int> threshold_3chika_otsu_flag_edit(int image_xt, int image_yt, double **Vt) {

	int count_positive_value = 0;
	int count_negative_value = 0;

	for (int i = 0; i < image_yt; i++) {
		for (int j = 0; j < image_xt; j++) {
			if (Vt[j][i] >= 0) {
				++count_positive_value;
			}
			else {
				++count_negative_value;
			}
		}
	}

	//printf("count_positive_value=%d,count_negative_value=%d\n", count_positive_value, count_negative_value);

	/*
	����
	std::tie(count_positive_value, count_negative_value)=threshold_3chika_otsu_flag_edit(image_xt, image_yt, V0t)
	*/
	return std::forward_as_tuple(count_positive_value, count_negative_value);
	//count_positive_value��count_negative_value��image_x(image_xt)�Ƃ��đ�Â�2�l���Ɏ�������,imagey=1
}

std::tuple< std::vector<std::vector<double>>, std::vector<std::vector<double>>> threshold_3chika_otsu_edit(int image_xt, int image_yt, double **Vt, int i1,int i2) {
//std::tuple< int, std::vector<std::vector<double>>> threshold_3chika_otsu_edit(int image_xt, int image_yt, double **Vt, int i1, int i2) {
	int i1_count = 0;
	int i2_count = 0;

	std::vector<std::vector<double>>Vt_positive;
	Vt_positive.resize(i1);
	for (int i = 0; i < i1; ++i) {
		Vt_positive[i].resize(1);
	}

	std::vector<std::vector<double>>Vt_negative;
	Vt_negative.resize(i2);
	for (int i = 0; i<i2; ++i) {
		Vt_negative[i].resize(1);
	}

	for (int i = 0; i < image_yt; i++) {
		for (int j = 0; j < image_xt; j++) {
			if (Vt[j][i] >= 0) {
				Vt_positive[i1_count][0] = Vt[j][i];
			//	printf("Vt_positive=%lf\n", Vt_positive[i1_count][0]);
				++i1_count;
				
			}
			else {
				Vt_negative[i2_count][0] = Vt[j][i]*-1;
				//printf("i2=%d\n", i2);
				++i2_count;
			}
		}
	}
	
	return std::forward_as_tuple(Vt_positive, Vt_negative);
	//return std::forward_as_tuple(i1_count, Vt_positive);

}