
//メモリ確保を行うためのヘッダ
#define ANSI				
#include "nrutil.h"	

#include<stdio.h>
#include<math.h>
#include <omp.h>	//openMP

#include<fstream>
#include<iostream>
#include<string>
#include<sstream> //文字ストリーム

#include<time.h>//時間を用いる

#include <direct.h>//フォルダを作成する

#include <opencv2/opencv.hpp>	//画像読み込み

#include<thread>	//複数スレッド

#include <tuple>

using namespace std;

//出力ファイル名・出力先の設定
char date_directoryC2[256];
char date_directoryC3[128];
char FilenameC[256];
char FilenameC1[256];
char FilenameC2[256];
char FilenameC3[256];
char FilenameC4[256];
char FilenameC5[256];
char FilenameC6[256];
char FilenameC7[256];
char FilenameC8[256];
char FilenameC11[256];
char FilenameC12[256];
char FilenameC13[256];
char FilenameC14[256];
char FilenameC15[256];
char FilenameC16[256];
char FilenameC17[256];
char FilenameC18[256];


int write_frame(char date_directory[], char Inputiamge[], std::vector<int> max_x, std::vector<int> max_y, int image_xt, int image_yt,int count_tied_V_vote,int V_vote_max);
void chika_bmp(char date_directory2[], char date_directory3[], int &image_x, int &image_y, int make_image_repeat,double **V_2chika);
int threshold_data_edit(int image_xt, int image_yt, double **threshold_edit, double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t, int use_convolution_direction_flag[]);
double edge_st_temp(char date_directory[], int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, double **edge_st);
std::tuple<int, int> threshold_3chika_otsu_flag_edit(int image_xt, int image_yt, double **Vt);
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> threshold_3chika_otsu_edit(int image_xt, int image_yt, double **Vt,int i1,int i2);
//std::tuple<int, std::vector<std::vector<double>>> threshold_3chika_otsu_edit(int image_xt, int image_yt, double **Vt, int i1, int i2);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////最大のV_voteとその座標を求める////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<int>, double, int> max_v_vote_calculate(double **V_vote,int image_x,int image_y,int image_xt,int image_yt) {

	std::vector<int> max_x;
	std::vector<int> max_y;
	int hajime_count = 0;
	double max_V = 0;
	int count_tied_V_vote = 0;
	max_V = V_vote[0][0];

	for (int y = 0; y < image_y - image_yt; y++) {
		for (int x = 0; x < image_x - image_xt; x++) {

			if (V_vote[x][y] > max_V) {
				hajime_count = 1;
				count_tied_V_vote = 1;
				max_x.resize(count_tied_V_vote);
				max_y.resize(count_tied_V_vote);

				max_V = V_vote[x][y];
				max_x[0] = x;
				max_y[0] = y;
				//printf("V[%d][%d]=%f,", x, y, V[x][y]);
			}
			if (V_vote[x][y] == max_V &&hajime_count != 1) {
				max_x.resize(count_tied_V_vote + 1);
				max_y.resize(count_tied_V_vote + 1);
				max_x[count_tied_V_vote] = x;
				max_y[count_tied_V_vote] = y;
				++count_tied_V_vote;
			}

			hajime_count = 0;
		}
	}

	hajime_count = 0;
	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, max_V);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////テンプレート画像の投票権の確認（最大投票権_M×N)////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Voting_rights_template_sum8(
	int M, int N, int Bs, double **CB_buf,
	double **threshold_flag_V0t, double **threshold_flag_V45t, double **threshold_flag_V90t, double **threshold_flag_V135t,
	double **threshold_flag_V180t, double **threshold_flag_V225t, double **threshold_flag_V270t, double **threshold_flag_V315t
) {
	//そのブロックに閾値を超えるブロックがないかを判断
	//CB_bufが1だとそのブロックが投票権をもつ．
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {

			int bm = Bs*m;
			int bn = Bs*n;
			for (int i = 0; i < Bs; i++) {
				for (int j = 0; j < Bs; j++) {

					if (threshold_flag_V0t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V45t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V90t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V135t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V180t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V225t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V270t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
					if (threshold_flag_V315t[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
				}
			}
			if (CB_buf[n][m] > 0)CB_buf[n][m] = 1;
			//	printf("%lf,", CB_buf[n][m]);
		}
		//printf("\n");
	}

	return **CB_buf;

}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////対象画像の投票権の確認（普通使わない）////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Voting_rights_sum8(
	int image_x, int image_y, double **threshold_flag_V,
	double **threshold_flag_V0, double **threshold_flag_V45, double **threshold_flag_V90, double **threshold_flag_V135,
	double **threshold_flag_V180, double **threshold_flag_V225, double **threshold_flag_V270, double **threshold_flag_V315
) {

	for (int i = 0; i < image_y; i++) {
		for (int j = 0; j < image_x; j++) {
			if (threshold_flag_V0[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V45[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V90[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V135[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V180[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V225[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V270[j][i] != 0)threshold_flag_V[j][i] += 1;
			if (threshold_flag_V315[j][i] != 0)threshold_flag_V[j][i] += 1;

			if (threshold_flag_V[j][i] > 0)threshold_flag_V[j][i] = 1;

			//	printf("%lf,", threshold_flag_V[j][i]);

		}
		//printf("\n");
	}
	return **threshold_flag_V;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////最大投票権 M×Nのマッチング/////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<std::vector<int>, std::vector<int>,int,int> vote_maching_sum8(int use_threshold,int use_convolution_direction_flag[], int frame_allowable_error, int Bs, int image_x, int image_y, int image_xt, int image_yt, int N, int M, double **CB, double **CB_buf, double **V_vote,
	double **threshold_flag_V0t, double **threshold_flag_V45t, double **threshold_flag_V90t, double **threshold_flag_V135t,
	double **threshold_flag_V180t, double **threshold_flag_V225t, double **threshold_flag_V270t, double **threshold_flag_V315t,
	double **threshold_flag_V0, double **threshold_flag_V45, double **threshold_flag_V90, double **threshold_flag_V135,
	double **threshold_flag_V180, double **threshold_flag_V225, double **threshold_flag_V270, double **threshold_flag_V315,
	double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t,
	double **V0, double **V45, double **V90, double **V135, double **V180, double **V225, double **V270, double **V315
) {

	double **V_vote2 = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	//初期化
	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			V_vote2[j][i] = 0;
		}
	}

	//特定のブロックについてのループ

	for (int n = 0; n < N; n++) {

		for (int m = 0; m < M; m++) {

			int bm = Bs*m;
			int bn = Bs*n;
			double min_CB = 0;
			int min_x = 0;
			int min_y = 0;
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {
					CB[x][y] = 0;
				}
			}
			//printf("\nimage_x=%d\nimage_xt=%d\n", image_x, image_xt);

			//ここで閾値を用いる判定
			//特定方向が用いるかどうかの判断
			switch (use_convolution_direction_flag[0]) {
			case 1:

			//探索対象画像に対するループ
#pragma omp parallel for num_threads(6)
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {

					//ブロック内に関するループ

					for (int k = 0; k < Bs; k++) {
						for (int l = 0; l < Bs; l++) {
								//ここで閾値を用いる判定
								if (use_threshold != 0) {

									if (use_convolution_direction_flag[1] == 1)if (abs(threshold_flag_V0t[bm + l][bn + k]) == 1)CB[x][y] += abs(V0[x + bm + l][y + bn + k] - V0t[bm + l][bn + k]);
									if (use_convolution_direction_flag[2] == 1)if (abs(threshold_flag_V45t[bm + l][bn + k]) == 1)CB[x][y] += abs(V45[x + bm + l][y + bn + k] - V45t[bm + l][bn + k]);
									if (use_convolution_direction_flag[3] == 1)if (abs(threshold_flag_V90t[bm + l][bn + k]) == 1)CB[x][y] += abs(V90[x + bm + l][y + bn + k] - V90t[bm + l][bn + k]);
									if (use_convolution_direction_flag[4] == 1)if (abs(threshold_flag_V135t[bm + l][bn + k]) == 1)CB[x][y] += abs(V135[x + bm + l][y + bn + k] - V135t[bm + l][bn + k]);
									if (use_convolution_direction_flag[5] == 1)if (abs(threshold_flag_V180t[bm + l][bn + k]) == 1)CB[x][y] += abs(V180[x + bm + l][y + bn + k] - V180t[bm + l][bn + k]);
									if (use_convolution_direction_flag[6] == 1)if (abs(threshold_flag_V225t[bm + l][bn + k]) == 1)CB[x][y] += abs(V225[x + bm + l][y + bn + k] - V225t[bm + l][bn + k]);
									if (use_convolution_direction_flag[7] == 1)if (abs(threshold_flag_V270t[bm + l][bn + k]) == 1)CB[x][y] += abs(V270[x + bm + l][y + bn + k] - V270t[bm + l][bn + k]);
									if (use_convolution_direction_flag[8] == 1)if (abs(threshold_flag_V315t[bm + l][bn + k]) == 1)CB[x][y] += abs(V315[x + bm + l][y + bn + k] - V315t[bm + l][bn + k]);

								}else {

									if (use_convolution_direction_flag[1] == 1)CB[x][y] += abs(V0[x + bm + l][y + bn + k] - V0t[bm + l][bn + k]);
									if (use_convolution_direction_flag[2] == 1)CB[x][y] += abs(V45[x + bm + l][y + bn + k] - V45t[bm + l][bn + k]);
									if (use_convolution_direction_flag[3] == 1)CB[x][y] += abs(V90[x + bm + l][y + bn + k] - V90t[bm + l][bn + k]);
									if (use_convolution_direction_flag[4] == 1)CB[x][y] += abs(V135[x + bm + l][y + bn + k] - V135t[bm + l][bn + k]);
									if (use_convolution_direction_flag[5] == 1)CB[x][y] += abs(V180[x + bm + l][y + bn + k] - V180t[bm + l][bn + k]);
									if (use_convolution_direction_flag[6] == 1)CB[x][y] += abs(V225[x + bm + l][y + bn + k] - V225t[bm + l][bn + k]);
									if (use_convolution_direction_flag[7] == 1)CB[x][y] += abs(V270[x + bm + l][y + bn + k] - V270t[bm + l][bn + k]);
									if (use_convolution_direction_flag[8] == 1)CB[x][y] += abs(V315[x + bm + l][y + bn + k] - V315t[bm + l][bn + k]);
								}
						}

					}

				}

			}//ここまででcase1のループ
			break;
			

			case 0:
			default:
			//探索対象画像に対するループ
#pragma omp parallel for num_threads(6)
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {

					//ブロック内に関するループ

					for (int k = 0; k < Bs; k++) {
						for (int l = 0; l < Bs; l++) {
								//ここで閾値を用いる判定
								if (use_threshold != 0) {

									if (abs(threshold_flag_V0t[bm + l][bn + k]) == 1)CB[x][y] += abs(V0[x + bm + l][y + bn + k] - V0t[bm + l][bn + k]);
									if (abs(threshold_flag_V45t[bm + l][bn + k]) == 1)CB[x][y] += abs(V45[x + bm + l][y + bn + k] - V45t[bm + l][bn + k]);
									if (abs(threshold_flag_V90t[bm + l][bn + k]) == 1)CB[x][y] += abs(V90[x + bm + l][y + bn + k] - V90t[bm + l][bn + k]);
									if (abs(threshold_flag_V135t[bm + l][bn + k]) == 1)CB[x][y] += abs(V135[x + bm + l][y + bn + k] - V135t[bm + l][bn + k]);
									if (abs(threshold_flag_V180t[bm + l][bn + k]) == 1)CB[x][y] += abs(V180[x + bm + l][y + bn + k] - V180t[bm + l][bn + k]);
									if (abs(threshold_flag_V225t[bm + l][bn + k]) == 1)CB[x][y] += abs(V225[x + bm + l][y + bn + k] - V225t[bm + l][bn + k]);
									if (abs(threshold_flag_V270t[bm + l][bn + k]) == 1)CB[x][y] += abs(V270[x + bm + l][y + bn + k] - V270t[bm + l][bn + k]);
									if (abs(threshold_flag_V315t[bm + l][bn + k]) == 1)CB[x][y] += abs(V315[x + bm + l][y + bn + k] - V315t[bm + l][bn + k]);

								}else {

									CB[x][y] += abs(V0[x + bm + l][y + bn + k] - V0t[bm + l][bn + k]);
									CB[x][y] += abs(V45[x + bm + l][y + bn + k] - V45t[bm + l][bn + k]);
									CB[x][y] += abs(V90[x + bm + l][y + bn + k] - V90t[bm + l][bn + k]);
									CB[x][y] += abs(V135[x + bm + l][y + bn + k] - V135t[bm + l][bn + k]);
									CB[x][y] += abs(V180[x + bm + l][y + bn + k] - V180t[bm + l][bn + k]);
									CB[x][y] += abs(V225[x + bm + l][y + bn + k] - V225t[bm + l][bn + k]);
									CB[x][y] += abs(V270[x + bm + l][y + bn + k] - V270t[bm + l][bn + k]);
									CB[x][y] += abs(V315[x + bm + l][y + bn + k] - V315t[bm + l][bn + k]);

								}
							}
						}
					}
				}//ここまでcase0,defaultのループ
			break;

			}

			//m,nについてCBが最小となるx,yを求める
			//左上からはじめないようにする（CB[0][0]が0だとうまくいかない可能性がある
			int start_x = 2, start_y = 2;
			/*
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {
					
					if (threshold_flag_V0[x][y] == 0
						&& threshold_flag_V45[x][y] == 0
						&& threshold_flag_V90[x][y] == 0
						&& threshold_flag_V135[x][y] == 0
						&& threshold_flag_V180[x][y] == 0
						&& threshold_flag_V225[x][y] == 0
						&& threshold_flag_V270[x][y] == 0
						&& threshold_flag_V315[x][y] == 0
						)

					{
						//一つたりとも閾値を超えない場合は何もしない
					}
					else {
						min_CB = abs(CB[x][y]);
						start_x = x;
						start_y = y;
						
						break;
					}
				}
				if (start_y == y)break;
			}
			*/
	
			//if (n == 0 && m == 4)printf("firxt_min_CB=%lf\n",  min_CB);
			//printf("start_x=%d\nstart_y=%d\n", start_x, start_y);
			//m,nについてCBが最小となるx,yを求める
			min_CB = abs(CB[7][7]);
			int CB_count = 0;
			int CB_count_max = 0;
			
#pragma omp parallel for num_threads(6)
			for (int y = start_y; y< image_y - image_yt- start_y; y++) {

				for (int x = start_x; x < image_x - image_xt- start_x; x++) {
					++CB_count_max;
					
					if (min_CB>abs(CB[x][y])  ) {
						
							min_CB = abs(CB[x][y]);
							min_x = x;
							min_y = y;
					}
					
					
				}
			}
			int boundary_conditions = 0;
			if (boundary_conditions == 1) {
				if (n > 0 && n < N - 1) {
					if (m > 0 && m < M - 1) {
					
			
					if (use_threshold != 0) {
						if (CB_buf[n][m] == 1) {
							V_vote[min_x][min_y] += 1;
	//						printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
						}else {
	//						printf("CB[%d][%d](%d,%d)=thresholdを満たさないため投票しない\n", n, m, min_x, min_y);
						}
					}else {
						V_vote[min_x][min_y] += 1;
	//					printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
					}

					}
				}
			}
			else {
				if (use_threshold != 0) {
					if (CB_buf[n][m] == 1) {
						V_vote[min_x][min_y] += 1;
	//					printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
					}else {
	//					printf("CB[%d][%d](%d,%d)=thresholdを満たさないため投票しない\n", n, m, min_x, min_y);
					}
				}else {
					V_vote[min_x][min_y] += 1;
	//				printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
				}

			}

		}
	}

	
	


	double max_V=0;
	int count_tied_V_vote = 0;
	
	std::vector<int>max_x;
	std::vector<int>max_y;
	//最大のV_voteの座標とその値を求める
	std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);
	
	
	//複数の枠を統合する
	//int frame_allowable_error = 5;
	if (frame_allowable_error != 0 && count_tied_V_vote != 1) {
		//全ブロックに対して
		//±frame_allowable_errorの範囲を取る
		for (int i = 0; i < count_tied_V_vote + 1; ++i) {
			for (int k = -frame_allowable_error; k < frame_allowable_error + 1; ++k) {
				for (int l = -frame_allowable_error; l < frame_allowable_error + 1; ++l) {
					if (max_y[i] + l >= 0 && max_y[i] + l < image_y - image_yt) {
						if (max_x[i] + k >= 0 && max_x[i] + k < image_x - image_xt) {
							V_vote2[max_x[i]][max_y[i]] += V_vote[max_x[i] + k][max_y[i] + l];
						}
					}
				}
			}
		}
		//v_voteに入れなおす
		for (int i = 0; i < image_y - image_yt; i++) {
			for (int j = 0; j < image_x - image_xt; j++) {
				V_vote[j][i] = V_vote2[j][i];
			}
		}
	

		free_matrix(V_vote2, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
		max_V = 0;
		count_tied_V_vote = 0;
		int hajime_count = 0;

		max_V = V_vote[0][0];
		max_x.resize(1);
		max_y.resize(1);
		std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);
		
		int max_x_average = 0;
		int max_y_average = 0;
		int count_tied_number = 0;
		if (count_tied_V_vote != 1) {
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average += max_x[i];
				max_y_average += max_y[i];
			}
			max_x_average = max_x_average / count_tied_V_vote;
			max_y_average = max_y_average / count_tied_V_vote;
			int max_x_average_con_min = max_x[0];
			int max_y_average_con_min = max_y[0];
			int max_x_average_con = max_x[0];
			int max_y_average_con = max_y[0];
			int correct_max_x = 0;
			int correct_max_y = 0;
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average_con = abs(max_x_average - max_x[i]);
				max_y_average_con = abs(max_y_average - max_y[i]);
				if (max_x_average_con_min < max_x_average_con) {
					max_x_average_con_min = max_x_average_con;
					correct_max_x = i;
				}
				if (max_y_average_con_min < max_y_average_con) {
					max_y_average_con_min = max_y_average_con;
					correct_max_y = i;
				}
			}


			for (int i = 0; i < count_tied_V_vote + 1; ++i) {

				//	if (max_x[i] - max_x[0] <= 2 * frame_allowable_error && max_y[i] - max_y[0] <= 2 * frame_allowable_error) {
				//		max_x[i] = max_x[0];
				//		max_y[i] = max_y[0];
				if (max_x[i] - max_x[correct_max_x] <= 2 * frame_allowable_error && max_y[i] - max_y[correct_max_y] <= 2 * frame_allowable_error) {
					max_x[i] = max_x[correct_max_x];
					max_y[i] = max_y[correct_max_y];
					++count_tied_number;
					//ここでcount_tied_V_voteを変更する
					if (i = count_tied_V_vote && count_tied_number == count_tied_V_vote+1) {
						count_tied_V_vote = 1;
						max_x.resize(count_tied_V_vote);
						max_y.resize(count_tied_V_vote);
					}
				}

			}
		}
	}


	printf("max_V=%lf\n",  max_V);
	int V_vote_max = max_V;
	

	//最終結果の保存
	FILE *fp_CB, *fp_V;
	//char math_name[128];
	char math_name2[128];
	//char *math_name_s = "CB.csv";
	char *math_name2_s = "V_vote.csv";
	//sprintf(math_name, "%s\\%s", date_directoryC2, math_name_s);
	sprintf(math_name2, "%s\\%s", date_directoryC3, math_name2_s);
	//if ((fp_CB = fopen(math_name, "w")) == NULL) { printf("入力エラー CB.csvが開けません\nFile_name : %s", math_name); exit(1); }
	if ((fp_V = fopen(math_name2, "w")) == NULL) { printf("入力エラー V.csvが開けません\nFile_name : %s", math_name2); exit(1); }

	for (int i = 0; i < image_y - image_yt; ++i) {
		for (int j = 0; j < image_x - image_xt; ++j) {
			//		fprintf(fp_CB, "%lf,", CB[j][i]);
			//		if (j == image_x - image_xt - 1) { fprintf(fp_CB, "\n"); }
			fprintf(fp_V, "%lf,", V_vote[j][i]);
			if (j == image_x - image_xt - 1) { fprintf(fp_V, "\n"); }
		}
	}

	//	fclose(fp_CB);
	fclose(fp_V);

	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, V_vote_max);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////最大投票権 M×Nのマッチングの制御部分///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<int>,int,int> Sum8_maching(int use_threshold, int use_convolution_direction_flag[],int frame_allowable_error,int Bs, int image_x, int image_y, int image_xt, int image_yt, int M, int N,
	double **threshold_flag_V0t, double **threshold_flag_V45t, double **threshold_flag_V90t, double **threshold_flag_V135t,
	double **threshold_flag_V180t, double **threshold_flag_V225t, double **threshold_flag_V270t, double **threshold_flag_V315t,
	double **threshold_flag_V0, double **threshold_flag_V45, double **threshold_flag_V90, double **threshold_flag_V135,
	double **threshold_flag_V180, double **threshold_flag_V225, double **threshold_flag_V270, double **threshold_flag_V315,
	double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t,
	double **V0, double **V45, double **V90, double **V135, double **V180, double **V225, double **V270, double **V315
) {
	/////////////////マッチング////////////////////////////////////////////////////	


	int bm = 0;
	int bn = 0;

	double **CB = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **V_vote = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);

	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			CB[j][i] = 0;
			V_vote[j][i] = 0;
		}
	}

	//ブロック内にthresholdが満たすかどうかを確認する.CB_bufが1だとVの投票を行う
	double **CB_buf = matrix(0, N - 1, 0, M - 1);
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {
			CB_buf[n][m] = 0;
		}
	}

	//CB_buf_ブロックの投票権の確認
	Voting_rights_template_sum8(
		M, N, Bs, CB_buf,
		threshold_flag_V0t, threshold_flag_V45t, threshold_flag_V90t, threshold_flag_V135t,
		threshold_flag_V180t, threshold_flag_V225t, threshold_flag_V270t, threshold_flag_V315t
	);

	//対象画像の閾値を設定（ふつう使わない）
	double **threshold_flag_V = matrix(0, image_x - 1, 0, image_y - 1);
	Voting_rights_sum8(
		image_x, image_y, threshold_flag_V,
		threshold_flag_V0, threshold_flag_V45, threshold_flag_V90, threshold_flag_V135,
		threshold_flag_V180, threshold_flag_V225, threshold_flag_V270, threshold_flag_V315
	);

	std::vector<int> max_x;
	std::vector<int> max_y;
	int count_tied_V_vote;
	int V_vote_max;

	//CB
	std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = vote_maching_sum8(use_threshold, use_convolution_direction_flag, frame_allowable_error,Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote,
		threshold_flag_V0t, threshold_flag_V45t, threshold_flag_V90t, threshold_flag_V135t,
		threshold_flag_V180t, threshold_flag_V225t, threshold_flag_V270t, threshold_flag_V315t,
		threshold_flag_V0, threshold_flag_V45, threshold_flag_V90, threshold_flag_V135,
		threshold_flag_V180, threshold_flag_V225, threshold_flag_V270, threshold_flag_V315,
		V0t, V45t, V90t, V135t, V180t, V225t, V270t, V315t,
		V0, V45, V90, V135, V180, V225, V270, V315
	);

	
	free_matrix(CB, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(V_vote, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(CB_buf, 0, N - 1, 0, M - 1);
	free_matrix(threshold_flag_V, 0, image_x - 1, 0, image_y - 1);

	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, V_vote_max);

}




























///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////テンプレート画像の投票権の確認（最大投票権_M×N×8//////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Voting_rights_each8(int image_x, int image_y, double **threshold_flag_V, double **threshold_flag_V_degree) {

	for (int i = 0; i < image_y; i++) {
		for (int j = 0; j < image_x; j++) {
			if (threshold_flag_V_degree[j][i] != 0)threshold_flag_V[j][i] += 1;
		}
	}
	return **threshold_flag_V;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////対象画像の投票権の確認（普通使わない）////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int Voting_rights_template_each8(int M, int N, int Bs, double **CB_buf, double **threshold_flag_Vt) {

	//そのブロックに閾値を超えるブロックがないかを判断
	//CB_bufが1だとそのブロックが投票権をもつ．
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {
			int bm = Bs*m;
			int bn = Bs*n;
			for (int i = 0; i < Bs; i++) {
				for (int j = 0; j < Bs; j++) {
					if (threshold_flag_Vt[j + bm][i + bn] != 0)CB_buf[n][m] += 1;
				}
			}
			if (CB_buf[n][m] > 0)CB_buf[n][m] = 1;
		}
	}
	return **CB_buf;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////最大投票権 M×N×8のマッチング//////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int vote_maching_each8(int use_threshold, int Bs, int image_x, int image_y, int image_xt, int image_yt, int N, int M, double **CB, double **CB_buf, double **V_vote_buf, double **threshold_flag_Vt, double **threshold_flag_V, double **Vt, double **V) {

	//特定のブロックについてのループ
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {

			int bm = Bs*m;
			int bn = Bs*n;
			double min_CB = 0;
			int min_x = 0;
			int min_y = 0;
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {
					CB[x][y] = 0;
				}
			}
			//探索対象画像に対するループ
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {

					//ブロック内に関するループ
					for (int k = 0; k < Bs; k++) {
						for (int l = 0; l < Bs; l++) {

							//ここで閾値を用いる判定
							if (use_threshold != 0) {
								if (abs(threshold_flag_Vt[bm + l][bn + k]) == 1)CB[x][y] += abs(V[x + bm + l][y + bn + k] - Vt[bm + l][bn + k]);
							}else {
								CB[x][y] += abs(V[x + bm + l][y + bn + k] - Vt[bm + l][bn + k]);
							}
						}
					}
				}
			}

			//m,nについてCBが最小となるx,yを求める
			//左上からはじめないようにする（CB[0][0]が0だとうまくいかない可能性がある
			int start_x = 2, start_y = 2;
			/*
			for (int y = 0; y < image_y - image_yt; y++) {
				for (int x = 0; x < image_x - image_xt; x++) {
					if (threshold_flag_V[x][y] == 0){
						//一つたりとも閾値を超えない場合は何もしない
					}else{
						min_CB = abs(CB[x][y]);
						start_x = x;
						start_y = y;
						break;
					}
				}
				if (start_y == y)break;
			}
			*/
			min_CB = CB[2][2];
			//m,nについてCBが最小となるx,yを求める
			int CB_count = 0;
			int CB_count_max = 0;

			for (int y = start_y; y< image_y - image_yt-2; y++) {
				for (int x = start_x; x < image_x - image_xt-2; x++) {
					++CB_count_max;
			/*		if (abs(CB[x][y]) < min_CB) {
						if (threshold_flag_V[x][y] == 0

							) {
							++CB_count;
						}
						else {*/
							if (min_CB>abs(CB[x][y]) ) {
							//	if (CB[x][y] != 0) {
							min_CB = abs(CB[x][y]);
							min_x = x;
							min_y = y;
							//	}
					//	}
					}
				}
			}

			if (use_threshold != 0) {
				if (CB_buf[n][m] == 1) {
					V_vote_buf[min_x][min_y] += 1;
	//				printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
				}
				else {
	//				printf("CB[%d][%d](%d,%d)=thresholdを満たさないため投票しない\n", n, m, min_x, min_y);
				}
			}
			else {
				V_vote_buf[min_x][min_y] += 1;
	//			printf("CB[%d][%d](%d,%d)=%lf\n", n, m, min_x, min_y, CB[min_x][min_y]);
			}
		}
	}
	return **V_vote_buf;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////最大投票権 M×N×8のマッチングの制御部分///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<int>,int,int> Each8_maching(int use_threshold, int use_convolution_direction_flag[],int frame_allowable_error,int Bs, int image_x, int image_y, int image_xt, int image_yt, int M, int N,
	double **threshold_flag_V0t, double **threshold_flag_V45t, double **threshold_flag_V90t, double **threshold_flag_V135t,
	double **threshold_flag_V180t, double **threshold_flag_V225t, double **threshold_flag_V270t, double **threshold_flag_V315t,
	double **threshold_flag_V0, double **threshold_flag_V45, double **threshold_flag_V90, double **threshold_flag_V135,
	double **threshold_flag_V180, double **threshold_flag_V225, double **threshold_flag_V270, double **threshold_flag_V315,
	double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t,
	double **V0, double **V45, double **V90, double **V135, double **V180, double **V225, double **V270, double **V315,
	int vote_patern
) {
	/////////////////マッチング////////////////////////////////////////////////////	

	//定義
	int bm = 0;
	int bn = 0;
	double **CB = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	
	double **V_vote = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **V_vote2 = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **V_vote_buf = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **threshold_flag_V = matrix(0, image_x - 1, 0, image_y - 1);
	//int max_x, max_y;

	
	//初期化
	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			V_vote[j][i] = 0;
			V_vote2[j][i] = 0;
		}
	}
	

	//ブロック内にthresholdが満たすかどうかを確認する.CB_bufが1だとVの投票を行う
	double **CB_buf = matrix(0, N - 1, 0, M - 1);
	
	/////////特定方向を用いる//////////////////////////////////////////////////////////////////////
	switch (use_convolution_direction_flag[0]) {
	case 1:

		for (int each8 = 0; each8 < 8; ++each8) {
			//初期化
			for (int i = 0; i < image_y - image_yt; i++) {
				for (int j = 0; j < image_x - image_xt; j++) {
					CB[j][i] = 0;
					V_vote_buf[j][i] = 0;
				}
			}
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < M; m++) {
					CB_buf[n][m] = 0;
				}
			}
			//特定方向のみに投票権を与える
			switch (each8) {
			case 0:
				if (use_convolution_direction_flag[1] == 1) {
					//CB_buf_ブロックの投票権の確認
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V0t);
					//対象画像の閾値を設定（ふつう使わない）
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V0);
					//CB
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V0t, threshold_flag_V0, V0t, V0);
				}
			case 1:
				if (use_convolution_direction_flag[2] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V45t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V45);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V45t, threshold_flag_V45, V45t, V45);
				}
				break;
			case 2:
				if (use_convolution_direction_flag[3] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V90t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V90);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V90t, threshold_flag_V90, V90t, V90);
				}
				break;
			case 3:
				if (use_convolution_direction_flag[4] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V135t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V135);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V135t, threshold_flag_V135, V135t, V135);
				}
				break;
			case 4:
				if (use_convolution_direction_flag[5] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V180t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V180);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V180t, threshold_flag_V180, V180t, V180);
				}
				break;
			case 5:
				if (use_convolution_direction_flag[6] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V225t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V225);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V225t, threshold_flag_V225, V225t, V225);
				}
				break;
			case 6:
				if (use_convolution_direction_flag[7] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V270t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V270);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V270t, threshold_flag_V270, V270t, V270);
				}
				break;
			case 7:
				if (use_convolution_direction_flag[8] == 1) {
					Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V315t);
					Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V315);
					vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V315t, threshold_flag_V315, V315t, V315);
				}
				break;
			default:
				break;
			}

			for (int i = 0; i < image_y - image_yt; i++) {
				for (int j = 0; j < image_x - image_xt; j++) {
					V_vote[j][i] += V_vote_buf[j][i];
				}
			}

		}
		break;

		//8方向全部を用いる
	case 0:
	default:
		for (int each8 = 0; each8 < 8; ++each8) {

			//初期化
			for (int i = 0; i < image_y - image_yt; i++) {
				for (int j = 0; j < image_x - image_xt; j++) {
					CB[j][i] = 0;
					V_vote_buf[j][i] = 0;
				}
			}
			for (int n = 0; n < N; n++) {
				for (int m = 0; m < M; m++) {
					CB_buf[n][m] = 0;
				}
			}
			if (vote_patern < 8) { printf("一つの畳み込み結果を用います\n"); break; }	//畳み込みを1つしか用いない場合はこのループから抜ける

			switch (each8) {
			case 0:
				//CB_buf_ブロックの投票権の確認
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V0t);
				//対象画像の閾値を設定（ふつう使わない）
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V0);
				//CB
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V0t, threshold_flag_V0, V0t, V0);
				break;
			case 1:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V45t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V45);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V45t, threshold_flag_V45, V45t, V45);
				break;
			case 2:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V90t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V90);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V90t, threshold_flag_V90, V90t, V90);
				break;
			case 3:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V135t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V135);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V135t, threshold_flag_V135, V135t, V135);
				break;
			case 4:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V180t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V180);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V180t, threshold_flag_V180, V180t, V180);
				break;
			case 5:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V225t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V225);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V225t, threshold_flag_V225, V225t, V225);
				break;
			case 6:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V270t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V270);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V270t, threshold_flag_V270, V270t, V270);
				break;
			case 7:
				Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_V315t);
				Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V315);
				vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_V315t, threshold_flag_V315, V315t, V315);
				break;
			default:
				printf("おかしい\n");
				break;
			}
			//v_voteの計算
			for (int i = 0; i < image_y - image_yt; i++) {
				for (int j = 0; j < image_x - image_xt; j++) {
					V_vote[j][i] += V_vote_buf[j][i];
				}
			}
			

		}
		break;
	}

	//Vの大小を判別
	double max_V = 0;
	int count_tied_V_vote = 0;

	std::vector<int>max_x;
	std::vector<int>max_y;
	//最大のV_voteの座標とその値を求める
	std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);


	//複数の枠を統合する
	//int frame_allowable_error = 5;
	if (frame_allowable_error != 0 && count_tied_V_vote!=1) {
		//全ブロックに対して
		//±frame_allowable_errorの範囲を取る
		for (int i = 0; i < count_tied_V_vote + 1; ++i) {
			for (int k = -frame_allowable_error; k < frame_allowable_error + 1; ++k) {
				for (int l = -frame_allowable_error; l < frame_allowable_error + 1; ++l) {
					if (max_y[i] + l >= 0 && max_y[i] + l < image_y - image_yt) {
						if (max_x[i] + k >= 0 && max_x[i] + k < image_x - image_xt) {
							
							V_vote2[max_x[i]][max_y[i]] += V_vote[max_x[i] + k][max_y[i] + l];
							
						}
					}
				}
			}
		}
		//v_voteに入れなおす
		for (int i = 0; i < image_y - image_yt; i++) {
			for (int j = 0; j < image_x - image_xt; j++) {
				V_vote[j][i] = V_vote2[j][i];
			}
		}


		free_matrix(V_vote2, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
		max_V = 0;
		count_tied_V_vote = 0;
		int hajime_count = 0;

		max_V = V_vote[0][0];
		max_x.resize(1);
		max_y.resize(1);
		std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);

		int max_x_average = 0;
		int max_y_average = 0;
		int count_tied_number = 0;
		if (count_tied_V_vote != 1) {
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average += max_x[i];
				max_y_average += max_y[i];
			}
			max_x_average = max_x_average / count_tied_V_vote;
			max_y_average = max_y_average / count_tied_V_vote;
			int max_x_average_con_min = max_x[0];
			int max_y_average_con_min = max_y[0];
			int max_x_average_con = max_x[0];
			int max_y_average_con = max_y[0];
			int correct_max_x = 0;
			int correct_max_y = 0;
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average_con =abs( max_x_average - max_x[i]);
				max_y_average_con =abs( max_y_average - max_y[i]);
				if (max_x_average_con_min < max_x_average_con) {
					max_x_average_con_min = max_x_average_con;
					correct_max_x = i;
				}
				if (max_y_average_con_min < max_y_average_con) {
					max_y_average_con_min = max_y_average_con;
					correct_max_y = i;
				}
			}
			

			for (int i = 0; i < count_tied_V_vote + 1; ++i) {

			//	if (max_x[i] - max_x[0] <= 2 * frame_allowable_error && max_y[i] - max_y[0] <= 2 * frame_allowable_error) {
			//		max_x[i] = max_x[0];
			//		max_y[i] = max_y[0];
				if (max_x[i] - max_x[correct_max_x] <= 2 * frame_allowable_error && max_y[i] - max_y[correct_max_y] <= 2 * frame_allowable_error) {
					max_x[i] = max_x[correct_max_x];
					max_y[i] = max_y[correct_max_y];
					++count_tied_number;
					//ここでcount_tied_V_voteを変更する
					if (i = count_tied_V_vote && count_tied_number == count_tied_V_vote + 1) {
						count_tied_V_vote = 1;
						max_x.resize(count_tied_V_vote);
						max_y.resize(count_tied_V_vote);
					}
				}

			}
		}
	}

	int V_vote_max = max_V;
	//最終結果の保存
	FILE *fp_CB, *fp_V;
	//char math_name[128];
	char math_name2[128];
	//char *math_name_s = "CB.csv";
	char *math_name2_s = "V_vote.csv";
	//sprintf(math_name, "%s\\%s", date_directoryC2, math_name_s);
	sprintf(math_name2, "%s\\%s", date_directoryC3, math_name2_s);
	//if ((fp_CB = fopen(math_name, "w")) == NULL) { printf("入力エラー CB.csvが開けません\nFile_name : %s", math_name); exit(1); }
	if ((fp_V = fopen(math_name2, "w")) == NULL) { printf("入力エラー V_vote.csvが開けません\nFile_name : %s", math_name2); exit(1); }

	for (int i = 0; i < image_y - image_yt; ++i) {
		for (int j = 0; j < image_x - image_xt; ++j) {
			//		fprintf(fp_CB, "%lf,", CB[j][i]);
			//		if (j == image_x - image_xt - 1) { fprintf(fp_CB, "\n"); }
			fprintf(fp_V, "%lf,", V_vote[j][i]);
			if (j == image_x - image_xt - 1) { fprintf(fp_V, "\n"); }
		}
	}

	//	fclose(fp_CB);
	fclose(fp_V);



	free_matrix(CB, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(V_vote, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(V_vote_buf, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(CB_buf, 0, N - 1, 0, M - 1);
	free_matrix(threshold_flag_V, 0, image_x - 1, 0, image_y - 1);
	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, V_vote_max);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////Vのゲインによって用いるVを変更するマッチングの制御部分///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<int>, int, int> Select8_maching(int use_threshold, int use_convolution_direction_flag[], int frame_allowable_error, int Bs, int image_x, int image_y, int image_xt, int image_yt, int M, int N,
	double **threshold_flag_V0t, double **threshold_flag_V45t, double **threshold_flag_V90t, double **threshold_flag_V135t,
	double **threshold_flag_V180t, double **threshold_flag_V225t, double **threshold_flag_V270t, double **threshold_flag_V315t,
	double **threshold_flag_V0, double **threshold_flag_V45, double **threshold_flag_V90, double **threshold_flag_V135,
	double **threshold_flag_V180, double **threshold_flag_V225, double **threshold_flag_V270, double **threshold_flag_V315,
	double **V0t, double **V45t, double **V90t, double **V135t, double **V180t, double **V225t, double **V270t, double **V315t,
	double **V0, double **V45, double **V90, double **V135, double **V180, double **V225, double **V270, double **V315,
	int vote_patern
) {
	/////////////////マッチング////////////////////////////////////////////////////	

	//定義
	int bm = 0;
	int bn = 0;
	double **CB = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);

	double **V_vote = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **V_vote2 = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **V_vote_buf = matrix(0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	double **threshold_flag_V = matrix(0, image_x - 1, 0, image_y - 1);
	double **CB_buf = matrix(0, N - 1, 0, M - 1);			//ブロック内にthresholdが満たすかどうかを確認する.CB_bufが1だとVの投票を行う
	//int max_x, max_y;

	//初期化
	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			CB[j][i] = 0;
			V_vote_buf[j][i] = 0;
		}
	}
	for (int n = 0; n < N; n++) {
		for (int m = 0; m < M; m++) {
			CB_buf[n][m] = 0;
		}
	}
	
	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			V_vote[j][i] = 0;
			V_vote2[j][i] = 0;
		}
	}

	//8方向のゲインを比較した後に用いる配列群
	double **V_All = matrix(0, image_x - 1, 0, image_y - 1);
	double **V_All_flag = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V_All = matrix(0, image_x - 1, 0, image_y - 1);
	double **Vt_All = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **Vt_All_flag = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_Vt_All = matrix(0, image_xt - 1, 0, image_yt - 1);
	int use_convolution_direction_value_flag[9] = { 0,0,0,0,0,0,0,0,0 };
	double V_All_compare[9];
	double V_All_flag_compare[9];
	double Vt_All_compare[9];
	double Vt_All_flag_compare[9];

	//初期化
	for (int i = 0; i < image_y; i++) {
		for (int j = 0; j < image_x; j++) {
			V_All[j][i] = 0;
			V_All_flag[j][i] = 0;
			threshold_flag_V_All[j][i] = 0;
		}
	}
	for (int i = 0; i < image_yt; i++) {
		for (int j = 0; j < image_xt; j++) {
			Vt_All[j][i] = 0;
			Vt_All_flag[j][i] = 0;
			threshold_flag_Vt_All[j][i] = 0;
		}
	}

	
	
//最大値の検索
	double max_V_All = 0;
	//探索対象画像を編集．用いる方向を制限している場合は，用いない方法は0として扱う．
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {
				for (int k = 0; k < 8; k++) { V_All_compare[k] = 0; V_All_flag_compare[k] = 0;}
				
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[1] == 1){V_All_compare[0] = abs(V0[j][i]); V_All_flag_compare[0] = threshold_flag_V0[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[2] == 1){V_All_compare[1] = abs(V45[j][i]); V_All_flag_compare[1] = threshold_flag_V45[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[3] == 1){V_All_compare[2] = abs(V90[j][i]); V_All_flag_compare[2] = threshold_flag_V90[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[4] == 1){V_All_compare[3] = abs(V135[j][i]); V_All_flag_compare[3] = threshold_flag_V135[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[5] == 1){V_All_compare[4] = abs(V180[j][i]); V_All_flag_compare[4] = threshold_flag_V180[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[6] == 1){V_All_compare[5] = abs(V225[j][i]); V_All_flag_compare[5] = threshold_flag_V225[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[7] == 1){V_All_compare[6] = abs(V270[j][i]); V_All_flag_compare[6] = threshold_flag_V270[j][i];}
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[8] == 1){V_All_compare[7] = abs(V315[j][i]); V_All_flag_compare[7] = threshold_flag_V315[j][i];}
				
				//座標(i,j)の最大のVを求める
				max_V_All = V_All_compare[0];
				
				V_All[j][i] = V_All_compare[0];
				V_All_flag[j][i] = 0;
				threshold_flag_V_All[j][i] = V_All_flag_compare[0];
				for (int k = 0; k < 8; ++k) {
					if (max_V_All < V_All_compare[k]) {
						max_V_All = V_All_compare[k];
						V_All[j][i] = V_All_compare[k];
						V_All_flag[j][i] = k;
						threshold_flag_V_All[j][i]= V_All_flag_compare[k];
					}
				}

			}
		}
		//テンプレート画像を変更．用いる方向を制限している場合は，用いない方法は0として扱う．
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {
				for (int k = 0; k < 8; k++) { Vt_All_compare[k] = 0; Vt_All_flag_compare[k] = 0; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[1] == 1) { Vt_All_compare[0] = abs(V0t[j][i]); Vt_All_flag_compare[0] = threshold_flag_V0t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[2] == 1) { Vt_All_compare[1] = abs(V45t[j][i]); Vt_All_flag_compare[1] = threshold_flag_V45t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[3] == 1) { Vt_All_compare[2] = abs(V90t[j][i]); Vt_All_flag_compare[2] = threshold_flag_V90t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[4] == 1) { Vt_All_compare[3] = abs(V135t[j][i]); Vt_All_flag_compare[3] = threshold_flag_V135t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[5] == 1) { Vt_All_compare[4] = abs(V180t[j][i]); Vt_All_flag_compare[4] = threshold_flag_V180t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[6] == 1) { Vt_All_compare[5] = abs(V225t[j][i]); Vt_All_flag_compare[5] = threshold_flag_V225t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[7] == 1) { Vt_All_compare[6] = abs(V270t[j][i]); Vt_All_flag_compare[6] = threshold_flag_V270t[j][i]; }
				if (use_convolution_direction_flag[0] == 0 || use_convolution_direction_flag[8] == 1) { Vt_All_compare[7] = abs(V315t[j][i]); Vt_All_flag_compare[7] = threshold_flag_V315t[j][i]; }

				//座標(i,j)の最大のVtを求める
				double max_Vt_All = Vt_All_compare[0];
				Vt_All_flag[j][i] = 0;
				for (int k = 1; k < 8; ++k) {
					if (max_Vt_All > Vt_All_compare[k]) {
						max_Vt_All = Vt_All_compare[k];
						Vt_All[j][i] = Vt_All_compare[k];
						Vt_All_flag[j][i] = k;
						threshold_flag_Vt_All[j][i] = Vt_All_flag_compare[k];
					}
				}

			}
		}
	
	
				
	//CB_buf_ブロックの投票権の確認
	Voting_rights_template_each8(M, N, Bs, CB_buf, threshold_flag_Vt_All);
	//対象画像の閾値を設定（ふつう使わない）
	Voting_rights_each8(image_x, image_y, threshold_flag_V, threshold_flag_V_All);
	//CB
	vote_maching_each8(use_threshold, Bs, image_x, image_y, image_xt, image_yt, N, M, CB, CB_buf, V_vote_buf, threshold_flag_Vt_All, threshold_flag_V_All, Vt_All, V_All);
				
			

	for (int i = 0; i < image_y - image_yt; i++) {
		for (int j = 0; j < image_x - image_xt; j++) {
			V_vote[j][i] += V_vote_buf[j][i];
		}
	}

		

	//Vの大小を判別
	double max_V = 0;
	int count_tied_V_vote = 0;

	std::vector<int>max_x;
	std::vector<int>max_y;
	//最大のV_voteの座標とその値を求める
	std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);

	printf("max_x=%d,max_y=%d\n", max_x, max_y);
	//複数の枠を統合する
	//int frame_allowable_error = 5;
	if (frame_allowable_error != 0 && count_tied_V_vote != 1) {
		//全ブロックに対して
		//±frame_allowable_errorの範囲を取る
		for (int i = 0; i < count_tied_V_vote + 1; ++i) {
			for (int k = -frame_allowable_error; k < frame_allowable_error + 1; ++k) {
				for (int l = -frame_allowable_error; l < frame_allowable_error + 1; ++l) {
					if (max_y[i] + l >= 0 && max_y[i] + l < image_y - image_yt) {
						if (max_x[i] + k >= 0 && max_x[i] + k < image_x - image_xt) {

							V_vote2[max_x[i]][max_y[i]] += V_vote[max_x[i] + k][max_y[i] + l];

						}
					}
				}
			}
		}
		//v_voteに入れなおす
		for (int i = 0; i < image_y - image_yt; i++) {
			for (int j = 0; j < image_x - image_xt; j++) {
				V_vote[j][i] = V_vote2[j][i];
			}
		}


		free_matrix(V_vote2, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
		max_V = 0;
		count_tied_V_vote = 0;
		int hajime_count = 0;

		max_V = V_vote[0][0];
		max_x.resize(1);
		max_y.resize(1);
		std::tie(max_x, max_y, count_tied_V_vote, max_V) = max_v_vote_calculate(V_vote, image_x, image_y, image_xt, image_yt);
		
		int max_x_average = 0;
		int max_y_average = 0;
		int count_tied_number = 0;
		if (count_tied_V_vote != 1) {
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average += max_x[i];
				max_y_average += max_y[i];
			}
			max_x_average = max_x_average / count_tied_V_vote;
			max_y_average = max_y_average / count_tied_V_vote;
			int max_x_average_con_min = max_x[0];
			int max_y_average_con_min = max_y[0];
			int max_x_average_con = max_x[0];
			int max_y_average_con = max_y[0];
			int correct_max_x = 0;
			int correct_max_y = 0;
			for (int i = 0; i < count_tied_V_vote + 1; ++i) {
				max_x_average_con = abs(max_x_average - max_x[i]);
				max_y_average_con = abs(max_y_average - max_y[i]);
				if (max_x_average_con_min < max_x_average_con) {
					max_x_average_con_min = max_x_average_con;
					correct_max_x = i;
				}
				if (max_y_average_con_min < max_y_average_con) {
					max_y_average_con_min = max_y_average_con;
					correct_max_y = i;
				}
			}
			

			for (int i = 0; i < count_tied_V_vote + 1; ++i) {

				//	if (max_x[i] - max_x[0] <= 2 * frame_allowable_error && max_y[i] - max_y[0] <= 2 * frame_allowable_error) {
				//		max_x[i] = max_x[0];
				//		max_y[i] = max_y[0];
				if (max_x[i] - max_x[correct_max_x] <= 2 * frame_allowable_error && max_y[i] - max_y[correct_max_y] <= 2 * frame_allowable_error) {
					max_x[i] = max_x[correct_max_x];
					max_y[i] = max_y[correct_max_y];
					++count_tied_number;
					//ここでcount_tied_V_voteを変更する
					if (i = count_tied_V_vote && count_tied_number == count_tied_V_vote + 1) {
						count_tied_V_vote = 1;
						max_x.resize(count_tied_V_vote);
						max_y.resize(count_tied_V_vote);
					}
				}

			}
		}
	}

	int V_vote_max = max_V;
	//最終結果の保存
	FILE *fp_CB, *fp_V;
	//char math_name[128];
	char math_name2[128];
	//char *math_name_s = "CB.csv";
	char *math_name2_s = "V_vote.csv";
	//sprintf(math_name, "%s\\%s", date_directoryC2, math_name_s);
	sprintf(math_name2, "%s\\%s", date_directoryC3, math_name2_s);
	//if ((fp_CB = fopen(math_name, "w")) == NULL) { printf("入力エラー CB.csvが開けません\nFile_name : %s", math_name); exit(1); }
	if ((fp_V = fopen(math_name2, "w")) == NULL) { printf("入力エラー V_vote.csvが開けません\nFile_name : %s", math_name2); exit(1); }

	for (int i = 0; i < image_y - image_yt; ++i) {
		for (int j = 0; j < image_x - image_xt; ++j) {
			//		fprintf(fp_CB, "%lf,", CB[j][i]);
			//		if (j == image_x - image_xt - 1) { fprintf(fp_CB, "\n"); }
			fprintf(fp_V, "%lf,", V_vote[j][i]);
			if (j == image_x - image_xt - 1) { fprintf(fp_V, "\n"); }
		}
	}

	//	fclose(fp_CB);
	fclose(fp_V);


	
	
	free_matrix(V_All, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V_All_flag, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V_All, 0, image_x - 1, 0, image_y - 1);
	free_matrix(Vt_All, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(Vt_All_flag, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_Vt_All, 0, image_xt - 1, 0, image_yt - 1);

	free_matrix(CB, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(V_vote, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(V_vote_buf, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	free_matrix(CB_buf, 0, N - 1, 0, M - 1);
	free_matrix(threshold_flag_V, 0, image_x - 1, 0, image_y - 1);
	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, V_vote_max);
}
























void set_convolutionfile(char date[], char date_directory[], int paramerter[], int paramerter_count, int sd) {

	//////////////////////////////inputディレクトリの設定//////////////////////////////////////////////////////////////////////////////////
	//inputのファイル名の指定
	char *FilenameC1_s = "V(0).csv";
	char *FilenameC2_s = "V(45).csv";
	char *FilenameC3_s = "V(90).csv";
	char *FilenameC4_s = "V(135).csv";
	char *FilenameC5_s = "V(180).csv";
	char *FilenameC6_s = "V(225).csv";
	char *FilenameC7_s = "V(270).csv";
	char *FilenameC8_s = "V(315).csv";
	char *FilenameC11_s = "V(0)t.csv";
	char *FilenameC12_s = "V(45)t.csv";
	char *FilenameC13_s = "V(90)t.csv";
	char *FilenameC14_s = "V(135)t.csv";
	char *FilenameC15_s = "V(180)t.csv";
	char *FilenameC16_s = "V(225)t.csv";
	char *FilenameC17_s = "V(270)t.csv";
	char *FilenameC18_s = "V(315)t.csv";


	//畳み込み結果を保存するフォルダの作成
	//フォルダ名は実行日時になる
	sprintf(date_directory, "..\\result_usa\\%s\\", date);
	if (_mkdir(date_directory) == 0) {
		printf("フォルダ %s を作成しました\n", date_directory);
	}
	else {
		printf("フォルダ作成に失敗しました。もしくは作成済みです\n");
	}

	if (paramerter[0] == 1 || paramerter[0] == 2) {
		sprintf(date_directoryC2, "%s%d×%dsobel_conv_sd%d\\", date_directory, paramerter[paramerter_count], paramerter[paramerter_count], sd);
	}
	else {
		sprintf(date_directoryC2, "%s%dk_conv_sd%d\\", date_directory, paramerter[paramerter_count], sd);
	}
	//Outputディレクトリの作成
	if (_mkdir(date_directoryC2) == 0) {
		printf("フォルダ %s を作成しました\n", date_directoryC2);
	}
	else {
		printf("フォルダ作成に失敗しました。\n");
	}
	

	//inputファイルのディレクトリ設定
	sprintf(FilenameC1, "%s%s", date_directoryC2, FilenameC1_s);
	sprintf(FilenameC2, "%s%s", date_directoryC2, FilenameC2_s);
	sprintf(FilenameC3, "%s%s", date_directoryC2, FilenameC3_s);
	sprintf(FilenameC4, "%s%s", date_directoryC2, FilenameC4_s);
	sprintf(FilenameC5, "%s%s", date_directoryC2, FilenameC5_s);
	sprintf(FilenameC6, "%s%s", date_directoryC2, FilenameC6_s);
	sprintf(FilenameC7, "%s%s", date_directoryC2, FilenameC7_s);
	sprintf(FilenameC8, "%s%s", date_directoryC2, FilenameC8_s);
	sprintf(FilenameC11, "%s%s", date_directoryC2, FilenameC11_s);
	sprintf(FilenameC12, "%s%s", date_directoryC2, FilenameC12_s);
	sprintf(FilenameC13, "%s%s", date_directoryC2, FilenameC13_s);
	sprintf(FilenameC14, "%s%s", date_directoryC2, FilenameC14_s);
	sprintf(FilenameC15, "%s%s", date_directoryC2, FilenameC15_s);
	sprintf(FilenameC16, "%s%s", date_directoryC2, FilenameC16_s);
	sprintf(FilenameC17, "%s%s", date_directoryC2, FilenameC17_s);
	sprintf(FilenameC18, "%s%s", date_directoryC2, FilenameC18_s);
	
	

}

void output_file(char date[], char date_directory[], int paramerter[], int paramerter_count, int sd) {

	//畳み込み結果を保存するフォルダの作成
	//フォルダ名は実行日時になる
	sprintf(date_directory, "..\\result_usa\\%s\\", date);
	if (_mkdir(date_directory) == 0) {
		printf("フォルダ %s を作成しました\n", date_directory);
	}
	else {
		printf("フォルダ作成に失敗しました。もしくは作成済みです\n");
	}

	if (paramerter[0] == 1 || paramerter[0] == 2) {
		sprintf(date_directoryC3, "%s%d×%dsobel_conv_sd%d_maching\\", date_directory, paramerter[paramerter_count], paramerter[paramerter_count], sd);
	}
	else {
		sprintf(date_directoryC3, "%s%dk_conv_sd%d_maching\\", date_directory, paramerter[paramerter_count], sd);
	}
	//Outputディレクトリの作成
	if (_mkdir(date_directoryC3) == 0) {
		printf("フォルダ %s を作成しました\n", date_directoryC3);
	}
	else {
		printf("フォルダ作成に失敗しました。\n");
	}

}













int use_convolution_setting(int i,int n[]) {

	//int n[10];
	int k=0;
	//printf("i=%d\n", i);
	//入力された数列を1文字ずつにする．
	for (int j = 1; j <= 100000000; j = j * 10) {
		n[k] = i / (100000000 / j);
	//	printf("%d\n", n[k]);
		i = i % (100000000 / j);
		++k;
	}
	/*
	//チェック
	for (k = 0; k < 8; ++k) {
		if (k == '\0') {
			printf("simulation_pattern[6]の値がおかしい\nsimulation_pattern[6]=%d\n", i);
			return -1;
		}
	}
	*/
	return *n;
}





//マッチング全体の設定
std::tuple<std::vector<int>, std::vector<int>, int, int>convolution_maching(int simulation_pattern[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char conv_date[], char output_date[], char date_directory[], double threshold_high, double threshold_low, char Inputiamge[]) {
//int convolution_maching(int simulation_pattern[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char conv_date[], char output_date[], char date_directory[], double threshold_high, double threshold_low, char Inputiamge[]) {

	//初期設定
	set_convolutionfile(conv_date, date_directory, paramerter, paramerter_count, sd);
	output_file(output_date, date_directory, paramerter, paramerter_count, sd);
	int high_grayscale = 255;
	int low_grayscale = 0;
	int backgroundacale = 128;
	int flag_high = 1;
	int flag_low = -1;
	int Bs = simulation_pattern[0];
	int use_2chika = simulation_pattern[2];
	int use_threshold = simulation_pattern[3];
	int vote_patern = simulation_pattern[4];
	int use_convolution = simulation_pattern[5];
	int frame_allowable_error = simulation_pattern[7];
	int frame_boundary_condition = simulation_pattern[8];
	int use_convolution_direction_flag[10];

	switch (use_2chika) {
	case 0:
		printf("畳み込み画像を用います\nuse_2chika=%d", use_2chika);
		break;
	case 2:
		printf("2値化画像を用います\nuse_2chika=%d", use_2chika);
		printf("上の閾値をthresholdとして用います\tthreshold=%lf", threshold_high);
		break;
	case 3:
		printf("3値化画像を用います\nuse_2chika=%d", use_2chika);
		printf("threshold_high=%lf\nthreshold_low=%lf\n", threshold_high, threshold_low);
		break;
	default:
		printf("use_2chika=simulation_pattern[1]=%dの値がおかしい\n", use_2chika);
		exit(0);
		break;
	}

	switch (use_threshold) {
	case 0:
		printf("閾値をマッチングに用いません\nuse_threshold=%d", use_threshold);
		break;
	case 1:
		printf("閾値をマッチングに用います\nuse_threshold=%d", use_threshold);
		break;
	case 2:
		printf("上の閾値のみをマッチングに用います\nuse_threshold=%d", use_threshold);
		break;
	case 3:
		printf("下の閾値のみをマッチングに用います\nuse_threshold=%d", use_threshold);
		break;
	case 4:
		printf("判別分析法により，エッジ強度の閾値をマッチングに用います\nuse_threshold=%d", use_threshold);
		break;
	case 5:
		printf("大津の2値化を用いて正負それぞれの閾値を用いて3値化画像を用います\nuse_threshold=%d", use_threshold);
		threshold_high = 0;
		threshold_low = 0;
		break;
	default:
		printf("use_threshold=simulation_pattern[2]=%dの値がおかしい\n", use_threshold);
		exit(0);
		break;
	}

	switch (vote_patern) {
	case 9:
		printf("投票数をブロック数M×Nにします\nvote_patern=%d\n", vote_patern);
		break;
	case 8:
		printf("投票数をブロック数M×N×8にします\nvote_patern=%d\n", vote_patern);
		break;
	case 10:
		printf("応答電圧によって投票方法を変更します\nvote_patern=%d\n", vote_patern);
		break;
	case 0:
	case 1:
	case 2:
	case 3:
	case 4:
	case 5:
	case 6:
	case 7:
		printf("一つの畳み込みデータをマッチングに用います\n");
		break;
	default:
		printf("vote_patern=simulation_pattern[3]=%dの値がおかしい\n", vote_patern);
		exit(0);
		break;
	}

	//特定方向を用いる場合のチェック
	use_convolution_setting(use_convolution, use_convolution_direction_flag);
	if (use_convolution_direction_flag[0] == 1) {
		
		printf("特定方向のみを用いる:use_convolution_direction_flag=\n");
		for (int i = 1; i < 9; ++i) {
			printf("V%d:%d,", 45 * (i - 1), use_convolution_direction_flag[i]);
		}
		printf("\n");
	}

	//枠をまとめる場合のチェック
	switch (frame_allowable_error) {
	case 0:
		printf("枠をまとめませんframe_allowable_error=%d\n", frame_allowable_error);
		break;
	default:
		printf("一定範囲の枠をまとめますframe_allowable_error=%d\n", frame_allowable_error);
		break;		
	}

	//教か条件の確認
	switch (frame_boundary_condition) {
	case 0:
		printf("境界条件を考慮しません\n");
		break;
	default:
		printf("上下左右から%d[pixel]だけ境界条件を考慮します\n", frame_boundary_condition);
	//	image_xt = image_xt - (2 * frame_boundary_condition);
	//	image_yt = image_yt - (2 * frame_boundary_condition);
		break;
	}
	

	printf("threshold_high=%lf\nthreshold_low=%lf\n", threshold_high, threshold_low);

	//読み込むV
	double **V0_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V45_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V90_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V135_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V180_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V225_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V270_buf = matrix(0, image_x - 1, 0, image_y - 1);
	double **V315_buf = matrix(0, image_x - 1, 0, image_y - 1);

	double **V0t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V45t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V90t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V135t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V180t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V225t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V270t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V315t_buf = matrix(0, image_xt - 1, 0, image_yt - 1);

	//閾値判定を行ったもの．0か1
	double **V0 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V45 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V90 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V135 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V180 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V225 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V270 = matrix(0, image_x - 1, 0, image_y - 1);
	double **V315 = matrix(0, image_x - 1, 0, image_y - 1);

	double **V0t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V45t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V90t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V135t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V180t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V225t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V270t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **V315t = matrix(0, image_xt - 1, 0, image_yt - 1);

	//閾値判定用
	double **threshold_flag_V0 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V45 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V90 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V135 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V180 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V225 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V270 = matrix(0, image_x - 1, 0, image_y - 1);
	double **threshold_flag_V315 = matrix(0, image_x - 1, 0, image_y - 1);

	double **threshold_flag_V0t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V45t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V90t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V135t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V180t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V225t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V270t = matrix(0, image_xt - 1, 0, image_yt - 1);
	double **threshold_flag_V315t = matrix(0, image_xt - 1, 0, image_yt - 1);

	double threshold_Vt_otsu[8][2];

	//確保したメモリを初期化する
	for (int i = 0; i < image_y; i++) {
		for (int j = 0; j < image_x; j++) {

			V0_buf[j][i] = 0;
			V45_buf[j][i] = 0;
			V90_buf[j][i] = 0;
			V135_buf[j][i] = 0;
			V180_buf[j][i] = 0;
			V225_buf[j][i] = 0;
			V270_buf[j][i] = 0;
			V315_buf[j][i] = 0;

			V0[j][i] = 0;
			V45[j][i] = 0;
			V90[j][i] = 0;
			V135[j][i] = 0;
			V180[j][i] = 0;
			V225[j][i] = 0;
			V270[j][i] = 0;
			V315[j][i] = 0;


			threshold_flag_V0[j][i] = 0;
			threshold_flag_V45[j][i] = 0;
			threshold_flag_V90[j][i] = 0;
			threshold_flag_V135[j][i] = 0;
			threshold_flag_V180[j][i] = 0;
			threshold_flag_V225[j][i] = 0;
			threshold_flag_V270[j][i] = 0;
			threshold_flag_V315[j][i] = 0;

		}
	}

	for (int i = 0; i < image_yt; i++) {
		for (int j = 0; j < image_xt; j++) {

			V0t_buf[j][i] = 0;
			V45t_buf[j][i] = 0;
			V90t_buf[j][i] = 0;
			V135t_buf[j][i] = 0;
			V180t_buf[j][i] = 0;
			V225t_buf[j][i] = 0;
			V270t_buf[j][i] = 0;
			V315t_buf[j][i] = 0;

			V0t[j][i] = 0;
			V45t[j][i] = 0;
			V90t[j][i] = 0;
			V135t[j][i] = 0;
			V180t[j][i] = 0;
			V225t[j][i] = 0;
			V270t[j][i] = 0;
			V315t[j][i] = 0;

			threshold_flag_V0t[j][i] = 0;
			threshold_flag_V45t[j][i] = 0;
			threshold_flag_V90t[j][i] = 0;
			threshold_flag_V135t[j][i] = 0;
			threshold_flag_V180t[j][i] = 0;
			threshold_flag_V225t[j][i] = 0;
			threshold_flag_V270t[j][i] = 0;
			threshold_flag_V315t[j][i] = 0;
		}
	}

	//Inputファイルを開く
	ifstream V_0(FilenameC1);
	ifstream V_45(FilenameC2);
	ifstream V_90(FilenameC3);
	ifstream V_135(FilenameC4);
	ifstream V_180(FilenameC5);
	ifstream V_225(FilenameC6);
	ifstream V_270(FilenameC7);
	ifstream V_315(FilenameC8);

	ifstream V_0t(FilenameC11);
	ifstream V_45t(FilenameC12);
	ifstream V_90t(FilenameC13);
	ifstream V_135t(FilenameC14);
	ifstream V_180t(FilenameC15);
	ifstream V_225t(FilenameC16);
	ifstream V_270t(FilenameC17);
	ifstream V_315t(FilenameC18);

	if (!V_0) { printf("入力エラー V(0).csvがありません_convolution_matching\nInput_Filename=%s\n", FilenameC1);  exit(0);}
	if (!V_45) { printf("入力エラー V(45).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC2); exit(0);}
	if (!V_90) { printf("入力エラー V(90).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC3); exit(0) ; }
	if (!V_135) { printf("入力エラー V(135).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC4); exit(0); }
	if (!V_180) { printf("入力エラー V(180).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC5); exit(0) ; }
	if (!V_225) { printf("入力エラー V(225).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC6); exit(0) ; }
	if (!V_270) { printf("入力エラー V(270).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC7); exit(0) ; }
	if (!V_315) { printf("入力エラー V(315).csvがありません_convolution_matching\nInput_Filename=%s", FilenameC8); exit(0) ; }

	if (!V_0t) { printf("入力エラー V(0)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC11); exit(0) ; }
	if (!V_45t) { printf("入力エラー V(45)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC12); exit(0) ; }
	if (!V_90t) { printf("入力エラー V(90)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC13); exit(0) ; }
	if (!V_135t) { printf("入力エラー V(135)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC14); exit(0) ; }
	if (!V_180t) { printf("入力エラー V(180)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC15); exit(0) ; }
	if (!V_225t) { printf("入力エラー V(225)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC16); exit(0) ; }
	if (!V_270t) { printf("入力エラー V(270)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC17); exit(0) ; }
	if (!V_315t) { printf("入力エラー V(315)t.csvがありません_convolution_matching\nInput_Filename=%s", FilenameC18); exit(0) ; }


	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////応答電圧のcsvの読み込み_探索対象画像////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int i = 1;
	printf("応答電圧を読み取ります\n");
	string str_0, str_45, str_90, str_135, str_180, str_225, str_270, str_315;
	int count_large = 0;
	while (getline(V_0, str_0)) {					//このループ内ですべてを行う
		int count_small = 0;			//初期化

///////////////いろいろ定義．ここでやらないといけない///////////////////////////////////////////////////////////////////////////
		string token_V_0;
		istringstream stream_V_0(str_0);

		getline(V_45, str_45);	string token_V_45;	istringstream stream_V_45(str_45);
		getline(V_90, str_90);	string token_V_90;	istringstream stream_V_90(str_90);
		getline(V_135, str_135);	string token_V_135;	istringstream stream_V_135(str_135);
		getline(V_180, str_180);	string token_V_180;	istringstream stream_V_180(str_180);
		getline(V_225, str_225);	string token_V_225;	istringstream stream_V_225(str_225);
		getline(V_270, str_270);	string token_V_270;	istringstream stream_V_270(str_270);
		getline(V_315, str_315);	string token_V_315;	istringstream stream_V_315(str_315);

		//////////////////////////配列に代入//////////////////////////////////////////////////////////////////////////////////////////////////

		while (getline(stream_V_0, token_V_0, ',')) {	//一行読み取る．V0_bufのみ，繰り返しの範囲指定に用いる
			double tmp_V_0 = stof(token_V_0);			//文字を数字に変換
			V0_buf[count_small][count_large] = tmp_V_0;				//配列に代入
			//V0[count_small][count_large] = Rvectormagni[1] * V0[count_small][count_large];


			getline(stream_V_45, token_V_45, ',');
			double tmp_V_45 = stof(token_V_45);
			V45_buf[count_small][count_large] = tmp_V_45;
			//V45[count_small][count_large] = Rvectormagni[2] * V45[count_small][count_large];


			getline(stream_V_90, token_V_90, ',');
			double tmp_V_90 = stof(token_V_90);
			V90_buf[count_small][count_large] = tmp_V_90;
			//V90[count_small][count_large] = Rvectormagni[3] * V90[count_small][count_large];


			getline(stream_V_135, token_V_135, ',');
			double tmp_V_135 = stof(token_V_135);
			V135_buf[count_small][count_large] = tmp_V_135;
			//V135[count_small][count_large] = Rvectormagni[4] * V135[count_small][count_large];


			getline(stream_V_180, token_V_180, ',');
			double tmp_V_180 = stof(token_V_180);
			V180_buf[count_small][count_large] = tmp_V_180;
			//V180[count_small][count_large] = Rvectormagni[5] * V180[count_small][count_large];


			getline(stream_V_225, token_V_225, ',');
			double tmp_V_225 = stof(token_V_225);
			V225_buf[count_small][count_large] = tmp_V_225;
			//V225[count_small][count_large] = Rvectormagni[6] * V225[count_small][count_large];


			getline(stream_V_270, token_V_270, ',');
			double tmp_V_270 = stof(token_V_270);
			V270_buf[count_small][count_large] = tmp_V_270;
			//V270[count_small][count_large] = Rvectormagni[7] * V270[count_small][count_large];


			getline(stream_V_315, token_V_315, ',');
			double tmp_V_315 = stof(token_V_315);
			V315_buf[count_small][count_large] = tmp_V_315;
			//V315[count_small][count_large] = Rvectormagni[8] * V315[count_small][count_large];


			++count_small;
		}
		++count_large;
	}





	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////応答電圧のcsvの読み込み_テンプレート画像////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (frame_boundary_condition == 0) {
		i = 1;
		printf("応答電圧を読み取ります\n");
		string str_0t, str_45t, str_90t, str_135t, str_180t, str_225t, str_270t, str_315t;
		count_large = 0;
		while (getline(V_0t, str_0t)) {					//このループ内ですべてを行う
			int count_small = 0;			//初期化

	///////////////いろいろ定義．ここでやらないといけない///////////////////////////////////////////////////////////////////////////
			string token_V_0t;
			istringstream stream_V_0t(str_0t);

			getline(V_45t, str_45t);	string token_V_45t;	istringstream stream_V_45t(str_45t);
			getline(V_90t, str_90t);	string token_V_90t;	istringstream stream_V_90t(str_90t);
			getline(V_135t, str_135t);	string token_V_135t;	istringstream stream_V_135t(str_135t);
			getline(V_180t, str_180t);	string token_V_180t;	istringstream stream_V_180t(str_180t);
			getline(V_225t, str_225t);	string token_V_225t;	istringstream stream_V_225t(str_225t);
			getline(V_270t, str_270t);	string token_V_270t;	istringstream stream_V_270t(str_270t);
			getline(V_315t, str_315t);	string token_V_315t;	istringstream stream_V_315t(str_315t);

			//////////////////////////配列に代入//////////////////////////////////////////////////////////////////////////////////////////////////

			while (getline(stream_V_0t, token_V_0t, ',')) {	//一行読み取る．V0_bufのみ，繰り返しの範囲指定に用いる

				getline(stream_V_45t, token_V_45t, ',');
				getline(stream_V_90t, token_V_90t, ',');
				getline(stream_V_135t, token_V_135t, ',');
				getline(stream_V_180t, token_V_180t, ',');
				getline(stream_V_225t, token_V_225t, ',');
				getline(stream_V_270t, token_V_270t, ',');
				getline(stream_V_315t, token_V_315t, ',');

				double tmp_V_0t = stof(token_V_0t);			//文字を数字に変換
				V0t_buf[count_small][count_large] = tmp_V_0t;				//配列に代入
				//V0[count_small][count_large] = Rvectormagni[1] * V0[count_small][count_large];

				double tmp_V_45t = stof(token_V_45t);
				V45t_buf[count_small][count_large] = tmp_V_45t;
				//V45[count_small][count_large] = Rvectormagni[2] * V45[count_small][count_large];

				double tmp_V_90t = stof(token_V_90t);
				V90t_buf[count_small][count_large] = tmp_V_90t;
				//V90[count_small][count_large] = Rvectormagni[3] * V90[count_small][count_large];

				double tmp_V_135t = stof(token_V_135t);
				V135t_buf[count_small][count_large] = tmp_V_135t;
				//V135[count_small][count_large] = Rvectormagni[4] * V135[count_small][count_large];

				double tmp_V_180t = stof(token_V_180t);
				V180t_buf[count_small][count_large] = tmp_V_180t;
				//V180[count_small][count_large] = Rvectormagni[5] * V180[count_small][count_large];

				double tmp_V_225t = stof(token_V_225t);
				V225t_buf[count_small][count_large] = tmp_V_225t;
				//V225[count_small][count_large] = Rvectormagni[6] * V225[count_small][count_large];

				double tmp_V_270t = stof(token_V_270t);
				V270t_buf[count_small][count_large] = tmp_V_270t;
				//V270[count_small][count_large] = Rvectormagni[7] * V270[count_small][count_large];

				double tmp_V_315t = stof(token_V_315t);
				V315t_buf[count_small][count_large] = tmp_V_315t;
				//V315[count_small][count_large] = Rvectormagni[8] * V315[count_small][count_large];


				++count_small;
			}
			++count_large;
		}
	}
	else {
		i = 1;
		printf("応答電圧を読み取ります\n");
		string str_0t, str_45t, str_90t, str_135t, str_180t, str_225t, str_270t, str_315t;
		count_large = 0;
		while (getline(V_0t, str_0t)) {					//このループ内ですべてを行う
			int count_small = 0;			//初期化

											///////////////いろいろ定義．ここでやらないといけない///////////////////////////////////////////////////////////////////////////
			string token_V_0t;
			istringstream stream_V_0t(str_0t);

			getline(V_45t, str_45t);	string token_V_45t;	istringstream stream_V_45t(str_45t);
			getline(V_90t, str_90t);	string token_V_90t;	istringstream stream_V_90t(str_90t);
			getline(V_135t, str_135t);	string token_V_135t;	istringstream stream_V_135t(str_135t);
			getline(V_180t, str_180t);	string token_V_180t;	istringstream stream_V_180t(str_180t);
			getline(V_225t, str_225t);	string token_V_225t;	istringstream stream_V_225t(str_225t);
			getline(V_270t, str_270t);	string token_V_270t;	istringstream stream_V_270t(str_270t);
			getline(V_315t, str_315t);	string token_V_315t;	istringstream stream_V_315t(str_315t);

			//////////////////////////配列に代入//////////////////////////////////////////////////////////////////////////////////////////////////

			while (getline(stream_V_0t, token_V_0t, ',')) {	//一行読み取る．V0_bufのみ，繰り返しの範囲指定に用いる

				getline(stream_V_45t, token_V_45t, ',');
				getline(stream_V_90t, token_V_90t, ',');
				getline(stream_V_135t, token_V_135t, ',');
				getline(stream_V_180t, token_V_180t, ',');
				getline(stream_V_225t, token_V_225t, ',');
				getline(stream_V_270t, token_V_270t, ',');
				getline(stream_V_315t, token_V_315t, ',');
				if (count_small >= frame_boundary_condition && count_large >= frame_boundary_condition) {

					if (count_small < image_xt+ frame_boundary_condition && count_large <image_yt+ frame_boundary_condition) {

						double tmp_V_0t = stof(token_V_0t);			//文字を数字に変換
						V0t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_0t;				//配列に代入
																					//V0[count_small][count_large] = Rvectormagni[1] * V0[count_small][count_large];

						double tmp_V_45t = stof(token_V_45t);
						V45t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_45t;
						//V45[count_small][count_large] = Rvectormagni[2] * V45[count_small][count_large];

						double tmp_V_90t = stof(token_V_90t);
						V90t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_90t;
						//V90[count_small][count_large] = Rvectormagni[3] * V90[count_small][count_large];

						double tmp_V_135t = stof(token_V_135t);
						V135t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_135t;
						//V135[count_small][count_large] = Rvectormagni[4] * V135[count_small][count_large];

						double tmp_V_180t = stof(token_V_180t);
						V180t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_180t;
						//V180[count_small][count_large] = Rvectormagni[5] * V180[count_small][count_large];

						double tmp_V_225t = stof(token_V_225t);
						V225t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_225t;
						//V225[count_small][count_large] = Rvectormagni[6] * V225[count_small][count_large];

						double tmp_V_270t = stof(token_V_270t);
						V270t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_270t;
						//V270[count_small][count_large] = Rvectormagni[7] * V270[count_small][count_large];

						double tmp_V_315t = stof(token_V_315t);
						V315t_buf[count_small - frame_boundary_condition][count_large - frame_boundary_condition] = tmp_V_315t;
						//V315[count_small][count_large] = Rvectormagni[8] * V315[count_small][count_large];

					}
				}
				++count_small;
			}
			++count_large;
		}

	}


	/////////////////////////////閾値判定/////////////////////////////////////////////////////////////////////
	double threshold_otsu = 0;		//大津の2値化を行う場合の閾値
	double **threshold_edit = matrix(0, image_xt - 1, 0, image_yt - 1);
	for (int i = 0; i <image_yt; i++) {
		for (int j = 0; j <image_xt; j++) {
			threshold_edit[j][i] = 0;
		}
	}

	int i1 = 0; int i2 = 0;//case 5:大津の2値化で正負それぞれ3値化するときに用いる
	int first_y = 1;
	//閾値を用いない場合でも，ここでは閾値を設定する．
	switch (use_threshold) {

		//上の閾値のみを用いる//////////////////////////////////////////////////////
	case 2:
		//探索対象画像
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {
				if (V0_buf[j][i] > threshold_high)threshold_flag_V0[j][i] = flag_high;
				if (V45_buf[j][i] > threshold_high)threshold_flag_V45[j][i] = flag_high;
				if (V90_buf[j][i] > threshold_high)threshold_flag_V90[j][i] = flag_high;
				if (V135_buf[j][i] > threshold_high)threshold_flag_V135[j][i] = flag_high;
				if (V180_buf[j][i] > threshold_high)threshold_flag_V180[j][i] = flag_high;
				if (V225_buf[j][i] > threshold_high)threshold_flag_V225[j][i] = flag_high;
				if (V270_buf[j][i] > threshold_high)threshold_flag_V270[j][i] = flag_high;
				if (V315_buf[j][i] > threshold_high)threshold_flag_V315[j][i] = flag_high;
			}
		}
		//テンプレート画像
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (V0t_buf[j][i] > threshold_high)threshold_flag_V0t[j][i] = flag_high;
				if (V45t_buf[j][i] > threshold_high)threshold_flag_V45t[j][i] = flag_high;
				if (V90t_buf[j][i] > threshold_high)threshold_flag_V90t[j][i] = flag_high;
				if (V135t_buf[j][i] > threshold_high)threshold_flag_V135t[j][i] = flag_high;
				if (V180t_buf[j][i] > threshold_high)threshold_flag_V180t[j][i] = flag_high;
				if (V225t_buf[j][i] > threshold_high)threshold_flag_V225t[j][i] = flag_high;
				if (V270t_buf[j][i] > threshold_high)threshold_flag_V270t[j][i] = flag_high;
				if (V315t_buf[j][i] > threshold_high)threshold_flag_V315t[j][i] = flag_high;

			}
		}
		break;

		//下の閾値のみを用いる//////////////////////////////////////////////////////
	case 3:
		//探索対象画像
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {
				if (V0_buf[j][i] < threshold_low)threshold_flag_V0[j][i] = flag_low;
				if (V45_buf[j][i] < threshold_low)threshold_flag_V45[j][i] = flag_low;
				if (V90_buf[j][i] < threshold_low)threshold_flag_V90[j][i] = flag_low;
				if (V135_buf[j][i] < threshold_low)threshold_flag_V135[j][i] = flag_low;
				if (V180_buf[j][i] < threshold_low)threshold_flag_V180[j][i] = flag_low;
				if (V225_buf[j][i] < threshold_low)threshold_flag_V225[j][i] = flag_low;
				if (V270_buf[j][i] < threshold_low)threshold_flag_V270[j][i] = flag_low;
				if (V315_buf[j][i] < threshold_low)threshold_flag_V315[j][i] = flag_low;
			}
		}
		//テンプレート画像
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (V0t_buf[j][i] < threshold_low)threshold_flag_V0t[j][i] = flag_low;
				if (V45t_buf[j][i] < threshold_low)threshold_flag_V45t[j][i] = flag_low;
				if (V90t_buf[j][i] < threshold_low)threshold_flag_V90t[j][i] = flag_low;
				if (V135t_buf[j][i] < threshold_low)threshold_flag_V135t[j][i] = flag_low;
				if (V180t_buf[j][i] < threshold_low)threshold_flag_V180t[j][i] = flag_low;
				if (V225t_buf[j][i] < threshold_low)threshold_flag_V225t[j][i] = flag_low;
				if (V270t_buf[j][i] < threshold_low)threshold_flag_V270t[j][i] = flag_low;
				if (V315t_buf[j][i] < threshold_low)threshold_flag_V315t[j][i] = flag_low;

			}
		}
		break;

		//エッジ強度を用いる//////////////////////////////////////////////////////
	case 4:
		//エッジ強度を求める
		
		threshold_data_edit(image_xt, image_yt, threshold_edit, V0t_buf, V45t_buf, V90t_buf, V135t_buf, V180t_buf, V225t_buf, V270t_buf, V315t_buf, use_convolution_direction_flag);
		printf("エッジ強度を用いるため注意\n");
		threshold_otsu = edge_st_temp(date_directoryC3, image_xt, image_yt, paramerter, paramerter_count, sd, threshold_edit);

		printf("threshold_otsu:b=%lf\n", threshold_otsu);

		
		//テンプレート画像
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (threshold_edit[j][i] > threshold_otsu) {
					threshold_flag_V0t[j][i] = flag_high;
					threshold_flag_V45t[j][i] = flag_high;
					threshold_flag_V90t[j][i] = flag_high;
					threshold_flag_V135t[j][i] = flag_high;
					threshold_flag_V180t[j][i] = flag_high;
					threshold_flag_V225t[j][i] = flag_high;
					threshold_flag_V270t[j][i] = flag_high;
					threshold_flag_V315t[j][i] = flag_high;
				}
				

			}
		}
		
		break;

		//大津の2値化的に正負それぞれに閾値を設定して3値化を行う
	case 5:
		

		//閾値判定
		for (int k = 0; k < 8; ++k) {
			double **Vt_3chika_threshold_buf = matrix(0, image_xt-1, 0, image_yt-1);
			for (int i = 0; i < image_yt; i++) {
				for (int j = 0; j < image_xt; j++) {
					Vt_3chika_threshold_buf[j][i] = 0;
				}
			}
			i1 = 0, i2 = 0;
			switch (k) {
			case 0:
				for (int i = 0; i < image_yt; i++) {for (int j = 0; j < image_xt; j++) {Vt_3chika_threshold_buf[j][i] = V0_buf[j][i];}}
				break;
			case 1:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V45_buf[j][i]; } }
				break;
			case 2:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V90_buf[j][i]; } }
				break;
			case 3:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V135_buf[j][i]; } }
				break;
			case 4:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V180_buf[j][i]; } }
				break;
			case 5:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V225_buf[j][i]; } }
				break;
			case 6:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V270_buf[j][i]; } }
				break;
			case 7:
				for (int i = 0; i < image_yt; i++) { for (int j = 0; j < image_xt; j++) { Vt_3chika_threshold_buf[j][i] = V315_buf[j][i]; } }
				break;
			default:
				break;
			}
			//正負それぞれの要素数を求める
			std::tie(i1, i2) = threshold_3chika_otsu_flag_edit(image_xt, image_yt, Vt_3chika_threshold_buf);
			printf("count_positive_value=%d,count_negative_value=%d\n", i1, i2);
			
			//配列の定義と初期化
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
			for (int j = 0; j < i1; j++) { Vt_positive[j][0] = 0; }
			for (int j = 0; j < i2; j++) { Vt_negative[j][0] = 0; }
			
			//正負それぞれを配列に代入する
			std::tie(Vt_positive, Vt_negative) = threshold_3chika_otsu_edit(image_xt, image_yt, Vt_3chika_threshold_buf,i1,i2);

			//配列の定義と初期化
			double **Vt_positive_N = matrix(0, i1 - 1, 0, 1);
			double **Vt_negative_N = matrix(0, i2 - 1, 0, 1);
			for (int j = 0; j < i1; j++) { Vt_positive_N[j][0] = 0; }
			for (int j = 0; j < i2; j++) { Vt_negative_N[j][0] = 0; }
			
			for (int j = 0; j < i1; j++) { Vt_positive_N[j][0] = Vt_positive[j][0]; 
			}
			for (int j = 0; j < i2; j++) { Vt_negative_N[j][0] = Vt_negative[j][0]; }
			
			threshold_high = edge_st_temp(date_directoryC3, i1, first_y, paramerter, paramerter_count, sd, Vt_positive_N);
			threshold_low = edge_st_temp(date_directoryC3, i2, first_y, paramerter, paramerter_count, sd, Vt_negative_N);
			threshold_low = -1 * threshold_low;

			printf("threshold_high=%lf,threshold_low=%lf,k=%d\n", threshold_high, threshold_low,k);
			//log用
			threshold_Vt_otsu[k][0] = threshold_high;
			threshold_Vt_otsu[k][1] = threshold_low;
			switch (k) {
			case 0:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V0_buf[j][i] > threshold_high)threshold_flag_V0[j][i] = flag_high;
						if (V0_buf[j][i] < threshold_low)threshold_flag_V0[j][i] = flag_low;
					}
				}
				break;

			case 1:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V45_buf[j][i] > threshold_high)threshold_flag_V45[j][i] = flag_high;
						if (V45_buf[j][i] < threshold_low)threshold_flag_V45[j][i] = flag_low;
					}
				}
				break;

			case 2:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V90_buf[j][i] > threshold_high)threshold_flag_V90[j][i] = flag_high;
						if (V90_buf[j][i] < threshold_low)threshold_flag_V90[j][i] = flag_low;
					}
				}
				break;

			case 3:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V135_buf[j][i] > threshold_high)threshold_flag_V135[j][i] = flag_high;
						if (V135_buf[j][i] < threshold_low)threshold_flag_V135[j][i] = flag_low;
					}
				}
				break;

			case 4:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V180_buf[j][i] > threshold_high)threshold_flag_V180[j][i] = flag_high;
						if (V180_buf[j][i] < threshold_low)threshold_flag_V180[j][i] = flag_low;
					}
				}
				break;

			case 5:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V225_buf[j][i] > threshold_high)threshold_flag_V225[j][i] = flag_high;
						if (V225_buf[j][i] < threshold_low)threshold_flag_V225[j][i] = flag_low;
					}
				}
				break;

			case 6:
				for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V270_buf[j][i] > threshold_high)threshold_flag_V270[j][i] = flag_high;
						if (V270_buf[j][i] < threshold_low)threshold_flag_V270[j][i] = flag_low;
					}
				}
				break;
			
			case 7:
			for (int i = 0; i < image_y; i++) {
					for (int j = 0; j < image_x; j++) {
						if (V315_buf[j][i] > threshold_high)threshold_flag_V315[j][i] = flag_high;
						if (V315_buf[j][i] < threshold_low)threshold_flag_V315[j][i] = flag_low;
					}
				}
			break;

			default:
				break;
			}
			//配列の解放
			free_matrix(Vt_positive_N, 0, i1 - 1, 0, 1);
			free_matrix(Vt_negative_N, 0, i2 - 1, 0, 1);
			free_matrix(Vt_3chika_threshold_buf, 0, image_xt - 1, 0, image_xt - 1);
		}


		
		

		//上下の閾値を用いる場合と，閾値を用いない場合//////////////////////////////////////////////////////
	default:
		//探索対象画像
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {

				if (V0_buf[j][i] > threshold_high)threshold_flag_V0[j][i] = flag_high;
				if (V45_buf[j][i] > threshold_high)threshold_flag_V45[j][i] = flag_high;
				if (V90_buf[j][i] > threshold_high)threshold_flag_V90[j][i] = flag_high;
				if (V135_buf[j][i] > threshold_high)threshold_flag_V135[j][i] = flag_high;
				if (V180_buf[j][i] > threshold_high)threshold_flag_V180[j][i] = flag_high;
				if (V225_buf[j][i] > threshold_high)threshold_flag_V225[j][i] = flag_high;
				if (V270_buf[j][i] > threshold_high)threshold_flag_V270[j][i] = flag_high;
				if (V315_buf[j][i] > threshold_high)threshold_flag_V315[j][i] = flag_high;

				if (V0_buf[j][i] < threshold_low)threshold_flag_V0[j][i] = flag_low;
				if (V45_buf[j][i] < threshold_low)threshold_flag_V45[j][i] = flag_low;
				if (V90_buf[j][i] < threshold_low)threshold_flag_V90[j][i] = flag_low;
				if (V135_buf[j][i] < threshold_low)threshold_flag_V135[j][i] = flag_low;
				if (V180_buf[j][i] < threshold_low)threshold_flag_V180[j][i] = flag_low;
				if (V225_buf[j][i] < threshold_low)threshold_flag_V225[j][i] = flag_low;
				if (V270_buf[j][i] < threshold_low)threshold_flag_V270[j][i] = flag_low;
				if (V315_buf[j][i] < threshold_low)threshold_flag_V315[j][i] = flag_low;
			}
		}
		//テンプレート画像
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (V0t_buf[j][i] > threshold_high)threshold_flag_V0t[j][i] = flag_high;
				if (V45t_buf[j][i] > threshold_high)threshold_flag_V45t[j][i] = flag_high;
				if (V90t_buf[j][i] > threshold_high)threshold_flag_V90t[j][i] = flag_high;
				if (V135t_buf[j][i] > threshold_high)threshold_flag_V135t[j][i] = flag_high;
				if (V180t_buf[j][i] > threshold_high)threshold_flag_V180t[j][i] = flag_high;
				if (V225t_buf[j][i] > threshold_high)threshold_flag_V225t[j][i] = flag_high;
				if (V270t_buf[j][i] > threshold_high)threshold_flag_V270t[j][i] = flag_high;
				if (V315t_buf[j][i] > threshold_high)threshold_flag_V315t[j][i] = flag_high;

				if (V0t_buf[j][i] < threshold_low)threshold_flag_V0t[j][i] = flag_low;
				if (V45t_buf[j][i] < threshold_low)threshold_flag_V45t[j][i] = flag_low;
				if (V90t_buf[j][i] < threshold_low)threshold_flag_V90t[j][i] = flag_low;
				if (V135t_buf[j][i] < threshold_low)threshold_flag_V135t[j][i] = flag_low;
				if (V180t_buf[j][i] < threshold_low)threshold_flag_V180t[j][i] = flag_low;
				if (V225t_buf[j][i] < threshold_low)threshold_flag_V225t[j][i] = flag_low;
				if (V270t_buf[j][i] < threshold_low)threshold_flag_V270t[j][i] = flag_low;
				if (V315t_buf[j][i] < threshold_low)threshold_flag_V315t[j][i] = flag_low;

			}
		}
		break;


	}

	///////////画素の設定_2値化の設定//////////////////////////////////////////////////////////////
	switch (use_2chika) {

		//生の応答電圧を用いる
	case 0:
		//実際の値を用いる
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {
				V0[j][i] = V0_buf[j][i];
				V45[j][i] = V45_buf[j][i];
				V90[j][i] = V90_buf[j][i];
				V135[j][i] = V135_buf[j][i];
				V180[j][i] = V180_buf[j][i];
				V225[j][i] = V225_buf[j][i];
				V270[j][i] = V270_buf[j][i];
				V315[j][i] = V315_buf[j][i];
			}
		}
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {
				V0t[j][i] = V0t_buf[j][i];
				V45t[j][i] = V45t_buf[j][i];
				V90t[j][i] = V90t_buf[j][i];
				V135t[j][i] = V135t_buf[j][i];
				V180t[j][i] = V180t_buf[j][i];
				V225t[j][i] = V225t_buf[j][i];
				V270t[j][i] = V270t_buf[j][i];
				V315t[j][i] = V315t_buf[j][i];

			}
		}
		break;
		//2値化を行う
	case 2:
		//画素の設定
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {

				if (threshold_flag_V0[j][i] == flag_high)V0[j][i] = high_grayscale;
				if (threshold_flag_V45[j][i] == flag_high)V45[j][i] = high_grayscale;
				if (threshold_flag_V90[j][i] == flag_high)V90[j][i] = high_grayscale;
				if (threshold_flag_V135[j][i] == flag_high)V135[j][i] = high_grayscale;
				if (threshold_flag_V180[j][i] == flag_high)V180[j][i] = high_grayscale;
				if (threshold_flag_V225[j][i] == flag_high)V225[j][i] = high_grayscale;
				if (threshold_flag_V270[j][i] == flag_high)V270[j][i] = high_grayscale;
				if (threshold_flag_V315[j][i] == flag_high)V315[j][i] = high_grayscale;

				if (threshold_flag_V0[j][i] == 0)V0[j][i] = backgroundacale;
				if (threshold_flag_V45[j][i] == 0)V45[j][i] = backgroundacale;
				if (threshold_flag_V90[j][i] == 0)V90[j][i] = backgroundacale;
				if (threshold_flag_V135[j][i] == 0)V135[j][i] = backgroundacale;
				if (threshold_flag_V180[j][i] == 0)V180[j][i] = backgroundacale;
				if (threshold_flag_V225[j][i] == 0)V225[j][i] = backgroundacale;
				if (threshold_flag_V270[j][i] == 0)V270[j][i] = backgroundacale;
				if (threshold_flag_V315[j][i] == 0)V315[j][i] = backgroundacale;
			}
		}
		//画素の設定
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (threshold_flag_V0t[j][i] == flag_high)V0t[j][i] = high_grayscale;
				if (threshold_flag_V45t[j][i] == flag_high)V45t[j][i] = high_grayscale;
				if (threshold_flag_V90t[j][i] == flag_high)V90t[j][i] = high_grayscale;
				if (threshold_flag_V135t[j][i] == flag_high)V135t[j][i] = high_grayscale;
				if (threshold_flag_V180t[j][i] == flag_high)V180t[j][i] = high_grayscale;
				if (threshold_flag_V225t[j][i] == flag_high)V225t[j][i] = high_grayscale;
				if (threshold_flag_V270t[j][i] == flag_high)V270t[j][i] = high_grayscale;
				if (threshold_flag_V315t[j][i] == flag_high)V315t[j][i] = high_grayscale;

				if (threshold_flag_V0t[j][i] == 0)V0t[j][i] = backgroundacale;
				if (threshold_flag_V45t[j][i] == 0)V45t[j][i] = backgroundacale;
				if (threshold_flag_V90t[j][i] == 0)V90t[j][i] = backgroundacale;
				if (threshold_flag_V135t[j][i] == 0)V135t[j][i] = backgroundacale;
				if (threshold_flag_V180t[j][i] == 0)V180t[j][i] = backgroundacale;
				if (threshold_flag_V225t[j][i] == 0)V225t[j][i] = backgroundacale;
				if (threshold_flag_V270t[j][i] == 0)V270t[j][i] = backgroundacale;
				if (threshold_flag_V315t[j][i] == 0)V315t[j][i] = backgroundacale;

			}
		}
		break;
		//3値化を行う
	case 3:
		//画素の設定
		for (int i = 0; i < image_y; i++) {
			for (int j = 0; j < image_x; j++) {

				if (threshold_flag_V0[j][i] == flag_high)V0[j][i] = high_grayscale;
				if (threshold_flag_V45[j][i] == flag_high)V45[j][i] = high_grayscale;
				if (threshold_flag_V90[j][i] == flag_high)V90[j][i] = high_grayscale;
				if (threshold_flag_V135[j][i] == flag_high)V135[j][i] = high_grayscale;
				if (threshold_flag_V180[j][i] == flag_high)V180[j][i] = high_grayscale;
				if (threshold_flag_V225[j][i] == flag_high)V225[j][i] = high_grayscale;
				if (threshold_flag_V270[j][i] == flag_high)V270[j][i] = high_grayscale;
				if (threshold_flag_V315[j][i] == flag_high)V315[j][i] = high_grayscale;

				if (threshold_flag_V0[j][i] == flag_low)V0[j][i] = low_grayscale;
				if (threshold_flag_V45[j][i] == flag_low)V45[j][i] = low_grayscale;
				if (threshold_flag_V90[j][i] == flag_low)V90[j][i] = low_grayscale;
				if (threshold_flag_V135[j][i] == flag_low)V135[j][i] = low_grayscale;
				if (threshold_flag_V180[j][i] == flag_low)V180[j][i] = low_grayscale;
				if (threshold_flag_V225[j][i] == flag_low)V225[j][i] = low_grayscale;
				if (threshold_flag_V270[j][i] == flag_low)V270[j][i] = low_grayscale;
				if (threshold_flag_V315[j][i] == flag_low)V315[j][i] = low_grayscale;

				if (threshold_flag_V0[j][i] == 0)V0[j][i] = backgroundacale;
				if (threshold_flag_V45[j][i] == 0)V45[j][i] = backgroundacale;
				if (threshold_flag_V90[j][i] == 0)V90[j][i] = backgroundacale;
				if (threshold_flag_V135[j][i] == 0)V135[j][i] = backgroundacale;
				if (threshold_flag_V180[j][i] == 0)V180[j][i] = backgroundacale;
				if (threshold_flag_V225[j][i] == 0)V225[j][i] = backgroundacale;
				if (threshold_flag_V270[j][i] == 0)V270[j][i] = backgroundacale;
				if (threshold_flag_V315[j][i] == 0)V315[j][i] = backgroundacale;
			}
		}
		//画素の設定
		for (int i = 0; i < image_yt; i++) {
			for (int j = 0; j < image_xt; j++) {

				if (threshold_flag_V0t[j][i] == flag_high)V0t[j][i] = high_grayscale;
				if (threshold_flag_V45t[j][i] == flag_high)V45t[j][i] = high_grayscale;
				if (threshold_flag_V90t[j][i] == flag_high)V90t[j][i] = high_grayscale;
				if (threshold_flag_V135t[j][i] == flag_high)V135t[j][i] = high_grayscale;
				if (threshold_flag_V180t[j][i] == flag_high)V180t[j][i] = high_grayscale;
				if (threshold_flag_V225t[j][i] == flag_high)V225t[j][i] = high_grayscale;
				if (threshold_flag_V270t[j][i] == flag_high)V270t[j][i] = high_grayscale;
				if (threshold_flag_V315t[j][i] == flag_high)V315t[j][i] = high_grayscale;

				if (threshold_flag_V0t[j][i] == flag_low)V0t[j][i] = low_grayscale;
				if (threshold_flag_V45t[j][i] == flag_low)V45t[j][i] = low_grayscale;
				if (threshold_flag_V90t[j][i] == flag_low)V90t[j][i] = low_grayscale;
				if (threshold_flag_V135t[j][i] == flag_low)V135t[j][i] = low_grayscale;
				if (threshold_flag_V180t[j][i] == flag_low)V180t[j][i] = low_grayscale;
				if (threshold_flag_V225t[j][i] == flag_low)V225t[j][i] = low_grayscale;
				if (threshold_flag_V270t[j][i] == flag_low)V270t[j][i] = low_grayscale;
				if (threshold_flag_V315t[j][i] == flag_low)V315t[j][i] = low_grayscale;

				if (threshold_flag_V0t[j][i] == 0)V0t[j][i] = backgroundacale;
				if (threshold_flag_V45t[j][i] == 0)V45t[j][i] = backgroundacale;
				if (threshold_flag_V90t[j][i] == 0)V90t[j][i] = backgroundacale;
				if (threshold_flag_V135t[j][i] == 0)V135t[j][i] = backgroundacale;
				if (threshold_flag_V180t[j][i] == 0)V180t[j][i] = backgroundacale;
				if (threshold_flag_V225t[j][i] == 0)V225t[j][i] = backgroundacale;
				if (threshold_flag_V270t[j][i] == 0)V270t[j][i] = backgroundacale;
				if (threshold_flag_V315t[j][i] == 0)V315t[j][i] = backgroundacale;

			}
		}
		break;
	
	}

	//////////////////////////////opencvを用いた2値化画像の作成///////////////////////////////////////////////////////////////////////////////////
	if (use_2chika != 0) {
		int make_image_repeat;

		for (make_image_repeat = 1; make_image_repeat <= 8; ++make_image_repeat) {
			switch (make_image_repeat) {
			case 1:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V0);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V0t);
				break;
			case 2:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V45);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V45t);
				break;
			case 3:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V90);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V90t);
				break;
			case 4:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V135);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V135t);
				break;
			case 5:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V180);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V180t);
				break;
			case 6:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V225);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V225t);
				break;
			case 7:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V270);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V270t);
				break;
			case 8:
				chika_bmp(date_directoryC2, date_directoryC3, image_x, image_y, make_image_repeat, V315);
				chika_bmp(date_directoryC2, date_directoryC3, image_xt, image_yt, make_image_repeat, V315t);
				break;
			}
		}

	}


	///////////////////////////////マッチング////////////////////////////////////////////////////

	std::vector<int> max_x;
	std::vector<int> max_y;//求める座標
	int count_tied_V_vote=0;
	int V_vote_max;
	int M, N;//ブロックの数

	M = image_xt / Bs;
	N = image_yt / Bs;

	printf("ブロックサイズBs=%d\n,M=%d,N=%d\n", Bs, M, N);

	//8方向の畳み込み結果を和として用いる
	if (vote_patern == 9) {	//8方向の和を用いる
		std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = Sum8_maching(use_threshold, use_convolution_direction_flag, frame_allowable_error, Bs, image_x, image_y, image_xt, image_yt, M, N,
			threshold_flag_V0t, threshold_flag_V45t, threshold_flag_V90t, threshold_flag_V135t,
			threshold_flag_V180t, threshold_flag_V225t, threshold_flag_V270t, threshold_flag_V315t,
			threshold_flag_V0, threshold_flag_V45, threshold_flag_V90, threshold_flag_V135,
			threshold_flag_V180, threshold_flag_V225, threshold_flag_V270, threshold_flag_V315,
			V0t, V45t, V90t, V135t, V180t, V225t, V270t, V315t,
			V0, V45, V90, V135, V180, V225, V270, V315
		);
	}

	if (vote_patern <= 8) {
		std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = Each8_maching(use_threshold, use_convolution_direction_flag, frame_allowable_error,Bs, image_x, image_y, image_xt, image_yt, M, N,
			threshold_flag_V0t, threshold_flag_V45t, threshold_flag_V90t, threshold_flag_V135t,
			threshold_flag_V180t, threshold_flag_V225t, threshold_flag_V270t, threshold_flag_V315t,
			threshold_flag_V0, threshold_flag_V45, threshold_flag_V90, threshold_flag_V135,
			threshold_flag_V180, threshold_flag_V225, threshold_flag_V270, threshold_flag_V315,
			V0t, V45t, V90t, V135t, V180t, V225t, V270t, V315t,
			V0, V45, V90, V135, V180, V225, V270, V315,
			vote_patern
		);

	}

	if (vote_patern == 10) {
		std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = Select8_maching(use_threshold, use_convolution_direction_flag, frame_allowable_error, Bs, image_x, image_y, image_xt, image_yt, M, N,
			threshold_flag_V0t, threshold_flag_V45t, threshold_flag_V90t, threshold_flag_V135t,
			threshold_flag_V180t, threshold_flag_V225t, threshold_flag_V270t, threshold_flag_V315t,
			threshold_flag_V0, threshold_flag_V45, threshold_flag_V90, threshold_flag_V135,
			threshold_flag_V180, threshold_flag_V225, threshold_flag_V270, threshold_flag_V315,
			V0t, V45t, V90t, V135t, V180t, V225t, V270t, V315t,
			V0, V45, V90, V135, V180, V225, V270, V315,
			vote_patern
		);

	}

	


	printf("count_tied_V_vote=%d\n", count_tied_V_vote);
	for (int i = 0; i < count_tied_V_vote; ++i) {
		printf("max_x[%d]=%d,max_y[%d]=%d\n", i, max_x[i], i, max_y[i]);

	}

	/////////////////////画像に枠を設置////////////////////////////////////////////////////////////////////////
	write_frame(date_directoryC3, Inputiamge, max_x, max_y, image_xt, image_yt, count_tied_V_vote, V_vote_max);

	//////////////////////////////logの作成///////////////////////////////////////////////////////////////////////////////////
	FILE *fp_date_Matching;
	char filename_log_Matching[128];
	//sprintf(filename_log, "..\\log\\log-%2d-%02d%02d-%02d%02d%02d.txt",pnow->tm_year+1900,pnow->tm_mon + 1,pnow->tm_mday,pnow->tm_hour,pnow->tm_min,pnow->tm_sec);	//logファイル作成のディレクトリ指定
	sprintf(filename_log_Matching, "%s\\log_Matching.txt", date_directoryC3);	//logファイル作成のディレクトリ指定
	if ((fp_date_Matching = fopen(filename_log_Matching, "w")) == NULL) { printf("logファイルが開けません"); exit(1); }
	fprintf(fp_date_Matching, "同率一位の数：count_tied_V_vote=%d(1でかぶりなし)\n", count_tied_V_vote);
	fprintf(fp_date_Matching, "最大の投票数：V_vote_max=%d\n", V_vote_max);
	for (int i = 0; i < count_tied_V_vote; ++i) {
		fprintf(fp_date_Matching, "領域の座標：(x,y)=(%d,%d),(%d,%d)\n", max_x[i], max_y[i], max_x[i] + image_xt, max_y[i] + image_yt);
	}

	fprintf(fp_date_Matching, "テンプレート画像サイズ：x=%d,y=%d\n", image_xt, image_yt);
	fprintf(fp_date_Matching, "ブロックサイズ：Bs=%d\n", Bs);
	fprintf(fp_date_Matching, "use_2chika=%d,use_threshold=%d\n", use_2chika, use_threshold);
	fprintf(fp_date_Matching, "vote_patern：%d\n", vote_patern);
	if (vote_patern == 9)fprintf(fp_date_Matching, "CBの投票数をブロック数とする(和を用いる)\n");
	if (vote_patern == 8)fprintf(fp_date_Matching, "CBの投票数をブロック数×8とする(それぞれ用いる)\n");
	if (vote_patern == 10)fprintf(fp_date_Matching, "CBの投票数をブロック数とする(Vのゲインで用いる値を決定するそれぞれ用いる)\n");
	if (use_convolution_direction_flag[0] != 1) {
		fprintf(fp_date_Matching, "CBやthresholdを全方向用いる\n");
	}
	else {
		fprintf(fp_date_Matching, "CBやthresholdを特定方向のみで求める\n");
		for (int i = 1; i < 9; ++i) {
			fprintf(fp_date_Matching, "V%d:%d,", 45 * (i - 1), use_convolution_direction_flag[i]);
		}
		fprintf(fp_date_Matching, "\n");
	}
		
		

	if (use_threshold == 5) {
		fprintf(fp_date_Matching, "threshold_Vt_otsu\n");
			for (int k = 0; k < 8; ++k) {
				
				fprintf(fp_date_Matching, "V%dt:high=%lf,low=%lf\n", 45*k, threshold_Vt_otsu[k][0], threshold_Vt_otsu[k][1]);
			}
	

	}

	if (use_threshold == 4)fprintf(fp_date_Matching, "判別分析法を用いる．threshold_otsu=%lf\n", threshold_otsu);
	
	if (frame_allowable_error != 0) { fprintf(fp_date_Matching, "近くの枠を統合する frame_allowable_error=%d\n", frame_allowable_error); }
	else {
		fprintf(fp_date_Matching, "素の枠を用いる frame_allowable_error=%d\n", frame_allowable_error);
	}

	//printf("max_x=%d\nmax_y=%d\n", max_x, max_y);

	fclose(fp_date_Matching);
	printf("logファイル %s を保存しました\n", filename_log_Matching);

	

	//メモリの開放
	free_matrix(V0_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V45_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V90_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V135_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V180_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V225_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V270_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V315_buf, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V0t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V45t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V90t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V135t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V180t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V225t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V270t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V315t_buf, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V0, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V45, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V90, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V135, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V180, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V225, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V270, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V315, 0, image_x - 1, 0, image_y - 1);
	free_matrix(V0t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V45t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V90t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V135t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V180t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V225t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V270t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(V315t, 0, image_xt - 1, 0, image_yt - 1);
	//free_matrix(CB, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);
	//free_matrix(V, 0, image_x - image_xt - 1, 0, image_y - image_yt - 1);

	
	free_matrix(threshold_flag_V0, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V45, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V90, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V135, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V180, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V225, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V270, 0, image_x - 1, 0, image_y - 1);
	free_matrix(threshold_flag_V315, 0, image_x - 1, 0, image_y - 1);

	free_matrix(threshold_flag_V0t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V45t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V90t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V135t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V180t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V225t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V270t, 0, image_xt - 1, 0, image_yt - 1);
	free_matrix(threshold_flag_V315t, 0, image_xt - 1, 0, image_yt - 1);


	free_matrix(threshold_edit, 0, image_xt - 1, 0, image_yt - 1);
	//free_matrix(CB_buf, 0, N - 1, 0, M - 1); 
	//free_matrix(threshold_flag_V, 0, image_x - 1, 0, image_y - 1);

	return std::forward_as_tuple(max_x, max_y, count_tied_V_vote, V_vote_max);
	//return 0;
}
