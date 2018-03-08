/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////関数の階層/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
	--main
		|--timeset																	//時間の取得
		|--or--convolution															//畳み込みを行う
		|		|--read_property													//畳み込みのpropertyを読み取る
		|		|--set_outputfile													//畳み込み結果を保存するファイル・フォルダの設定
		|		|--read_filter														//畳み込みに用いるフィルタを読み込む
		|		|--or--convolution_calculation										//探索対象画像でカーネルを用いた畳み込みを行う場合
		|		|  or--convolution_gaus_sobel										//探索対象画像でsobelフィルタかつgausフィルタを用いる場合
		|		|		|--read_filter_gaus											//探索対象画像で畳み込みに用いるフィルタと同等のサイズのgausフィルタを読み込む
		|		|--write_file														//探索対象画像で畳み込み結果のcsvファイル出力			
		|		|
		|		|--or--convolution_calculation										//template画像でカーネルを用いた畳み込みを行う場合
		|		|  or--convolution_gaus_sobel										//template画像でsobelフィルタかつgausフィルタを用いる場合
		|		|		|--read_filter_gaus											//template画像で畳み込みに用いるフィルタと同等のサイズのgausフィルタを読み込む
		|		|--write_file														//template画像で畳み込み結果のcsvファイル出食
		|		|
		|		|--make_bmp				×2×8										//畳み込み画像の作成
		|
		|	--or--read_log															//畳み込まずに，すでに畳み込みこまれたデータを用いる													
		|
		|--convolution_maching														//畳み込み画像を用いたブロックマッチング
		|	|--set_convolutionfile													//畳み込みファイルの読み込み	
		|	|--output_file															//outputファイルの設定
		|	|--threshold_data_edit													//thresholdに用いる配列が独特の場合はここで設定する
		|	|--edge_st_temp															//判別分析法（大津の2値化）で閾値を求める	
		|	|	|--discriminantAnalysis												//判別分析法（大津の2値化）で閾値を求める
		|	|		|--hist_hozon													//判別分析法で閾値を求める際に用いたヒストグラムを保存する
		|	|
		|	|--chika_bmp															//2値化画像の作成
		|	|--or--Sum8_maching														//CBを8方向合計する．つまり，最大投票件数はブロック数と等しい
		|	|		|--Voting_rights_template_sum8									//テンプレートについて投票権を確認する
		|	|		|--Voting_rights_sum8											//探索対象画像について投票権を確認する
		|	|		|--vote_maching_sum8											//マッチングを行う
		|	|			|--max_v_vote_calculate										//投票数の最大値を求める
		|	|
		|	|  or--Each8_maching													//CBを8方向それぞれで求める．つまり，最大投票件数はブロック数×8．ただし，畳み込み結果1つのみを用いる場合もここ
		|	|		|--Voting_rights_template_each8									//テンプレートについて投票権を確認する
		|	|		|--Voting_rights_each8											//探索対象画像について投票権を確認する
		|	|		|--vote_maching_each8											//マッチングを行う
		|	|			|--max_v_vote_calculate										//投票数の最大値を求める
		|	|
		|	|	or--Select8_maching													//各座標についてVのゲインを比較し，最も大きな値を用いてCBを求める．つまり，最大投票件数はブロック数と等しい．以下はEach8_machingと同じ
		|	|		|--Voting_rights_template_each8									//テンプレートについて投票権を確認する
		|	|		|--Voting_rights_each8											//探索対象画像について投票権を確認する
		|	|		|--vote_maching_each8											//マッチングを行う
		|	|			|--max_v_vote_calculate										//投票数の最大値を求める
		|	|
		|	|--write_frame															//探索対象画像にマッチング結果のフレームを記入する
		|
		|--score																	//統計処理を行うため，得点を計算する
		|	|--read_correct_xy														//統計処理のため，正しい位置のデータセットを読み込む
		|
		|--score_record																//得点を記録する

*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include<fstream>
#include<iostream>
#include<string>
#include<sstream>	//文字ストリーム
#include <direct.h>	//フォルダを作成する
#include<thread>	//複数スレッド
#include<vector>

using namespace std;

int image_x, image_y;		//画像サイズ
int image_xt, image_yt;		//画像サイズ

char date[128] = "";
char date_buf[128] = "";
//出力ファイルディレクトリ
char date_directory[128];
char image_nameP[128];
char image_nameP2[256];
int sd;

char Inputimage[128];



//kernelのパラメータとsobelのサイズを記す
int paramerter_kernel[4] = { 1,3,100,100 };
int paramerter_sobel[4] = { 0,3,5,7 };

int msectime();//msec単位の時間取得

int timeset(char date[]);
//int notimeset(char date[], int pixel[], int Togire[], int z2, int z);
int convolution(int argc, char** argv, char image_nameP2[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char date[], char date_directory[], char Inputimage[]);

std::tuple<std::vector<int>, std::vector<int>, int, int>convolution_maching(int simulation_pattern[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char conv_date[], char output_date[], char date_directory[], double threshold, double threshold_low, char Inputiamge[]);
//int convolution_maching(int simulation_pattern[],int &image_x, int &image_y,  int &image_xt, int &image_yt,int paramerter[], int paramerter_count, int sd, char conv_date[],char output_date[], char date_directory[], double threshold,  double threshold_low, char Inputiamge[]);

std::tuple<char&, int, int, int, int> read_log(int paramerter[], int paramerter_count, int sd, char date[], char Inputiamge[]);
int score(int correct_x, int correct_y, int image_xt, int image_yt, std::vector<int> max_x, std::vector<int> max_y, int count_tied_V_vote, int V_vote_max, int data_num, char correct_scv[], int high_score_range_x, int high_score_range_y);
int score_record(char date_directory[], int data_num,char Inputimage[], int correct_score, int count_tied_V_vote, int V_vote_max, std::vector<int> max_x, std::vector<int> max_y);



int main(int argc, char** argv) {

	//最終的に求めたい答え
	std::vector<int>max_x;
	std::vector<int>max_y;
	int count_tied_V_vote;
	int V_vote_max;


	//int pixel[10]={0,1,3,5,7,9,13,17};
	//int Togire[10] = { 0,1,3,5,7,9,13,17 };

	int paramerter[4];					//paramerter[0]=1でsobelフィルタ,paramerter[0]=2でgaus×sobelフィルタ
	paramerter[0] = 0;					//paramerter[0]=1でsobelフィルタ,paramerter[0]=2でgaus×sobelフィルタ

	double threshold = 1;
	double threshold_low = threshold*-1;
	//double threshold_low = -3;



	int simulation_pattern[9] = { 8, 0,0,0,9,011110000,180,5,20 };
	int sd_buf = 0;
	//手法を区別するためにsimulation_pattern[0]を変更する
	/*
							|		0			|		1			|		2		|		3		|		4				|		5			|
	ブロックサイズBs[0]		|		8																|
	畳み込むかどうか：[1]	|	畳み込みあり	|	畳み込みなし	|
	用いる2値化等	:[2]	|	2値化なし		|					|	2値化		|	3値化		|
	thresholdを用いる:[3]	|		なし		|		あり		|上の閾値のみ	|下の閾値のみ	|大津の2値化,エッジ強度|大津の2値化的な3値化|
	投票方法		:[4]	|	---																	|8:ブロック数×8		|	9:ブロック数	|10:応答電圧によって用いる部分を変更する|
	用いる方向		:[5]	|特定方向を用いるなら1|1~8:用いる方向
	統計処理を行う	:[6]	|	行わない		|行う，かつデータの数|
	枠をまとめる	:[7]	|まとめない			|	許容するズレを入れる
	投票権剥奪		:[8]	|左上からこの値だけ投票権を剥奪する．畳み込みは全画素に行う(例：テンプレート画像120×120で20に設定→80×80のpixelの畳み込み画像を投票に用いる)
	*/


	//標準偏差の調整箇所
	int sd_max = 50;
	int paramerter_count_max = 1;


	//正解座標の誤差
	int high_score_range_x = 0;
	int high_score_range_y = 0;
	high_score_range_x = simulation_pattern[7];
	high_score_range_y = simulation_pattern[7];

	//畳み込みの制御
	int no_convolution_frag = simulation_pattern[1];		//1だと畳み込みを行わない．
	int data_num = simulation_pattern[6];					//データを読み取るとき，データの数
	char no_date[128];										//出力フォルダを時刻ではなく個別に設定したいとき．用いるデータの階層に制限があるため注意
	//char no_date_directory_buf[128] = "simulation18-0115\\";//出力フォルダを時刻ではなく個別に設定したいとき．用いるデータの階層に制限があるため注意
	char no_date_directory_buf[128];
	//char *no_date_directory_buf = "simulation18-0115_sobel\\";//出力フォルダを時刻ではなく個別に設定したいとき．用いるデータの階層に制限があるため注意
	//sprintf(no_date, "convolution\\gpu2");

	//
	char *correct_scv = "..\\bmp\\simulation18-0115\\correct.csv";		//用いる探索正解座標

	int paramerter_count = 0;									//用いるパラメータの番号

	//用いるパラメータを代入
	switch (paramerter[0]) {
	case 0:
		for (int i = 1; i < 4; ++i) {
			paramerter[i] = paramerter_kernel[i];
		}
		break;
	case 1:
	case 2:
		for (int i = 1; i < 4; ++i) {
			paramerter[i] = paramerter_sobel[i];
		}
		break;
	default:
		printf("paramerter[0]の値がおかしい\nparamerter[0]=%d\n", paramerter[0]);
		return 0;
	}



//		for (sd = 0; sd <= sd_max; sd = sd + 10) {
			sprintf(no_date_directory_buf, "simulation18-0115_sd%d\\", sd);
	for (int image_kurikaeshi = 1; image_kurikaeshi < data_num + 1; ++image_kurikaeshi) {
		//	int image_kurikaeshi = 1;
		
		if (no_convolution_frag == 1)sprintf(no_date, "%s%d", no_date_directory_buf, image_kurikaeshi);
		//if (no_convolution_frag == 1)sprintf(no_date, "%s93", no_date_directory_buf);

		timeset(date);									//時間の取得
//		sprintf(date, "sd%d\\%d", sd, image_kurikaeshi);			//出力フォルダを時刻ではなく個別に設定したいとき．用いるデータの階層に制限があるため注意
		sprintf(date, "%d", image_kurikaeshi);			//出力フォルダを時刻ではなく個別に設定したいとき．用いるデータの階層に制限があるため注意
		//sprintf(date, "93");

		for (int paramerter_count = 1; paramerter_count <= paramerter_count_max; ++paramerter_count) {
			//	for (int sim_num = 1; sim_num < 4; ++sim_num) {

			for (sd = sd_buf; sd <= sd_buf; sd = sd + 10) {
				//			for (sd = 10; sd <= sd_max; sd = sd + 10) {

								//sobelフィルタを用いる時の設定
				if (paramerter[0] == 1 || paramerter[0] == 2) {

					sprintf(image_nameP, "..\\property_usa\\simulation18-0115\\property_%d×%dsobel_conv_", paramerter[paramerter_count], paramerter[paramerter_count]);
					sprintf(image_nameP2, "%ssd%d_%d.txt", image_nameP, sd, image_kurikaeshi);

					//kernelを用いる時の設定
				}
				else {

					//sprintf(image_nameP, "..\\property_usa\\simulation18-0111_hide_kernel\\property_%dk_conv_", paramerter[paramerter_count]);
					sprintf(image_nameP, "..\\property_usa\\simulation18-0115\\property_%dk_conv_", paramerter[paramerter_count]);
					//	sprintf(image_nameP, "..\\property_usa\\simulation18-0125\\property_%dk_conv_", paramerter[paramerter_count]);
					//	sprintf(image_nameP2, "%ssd%d_3.txt", image_nameP, sd);
					sprintf(image_nameP2, "%ssd%d_%d.txt", image_nameP, sd, image_kurikaeshi);
					//	sprintf(image_nameP2, "%ssd%d_5.txt", image_nameP, sd);
					//	sprintf(image_nameP2, "%ssd0_2_sd%d.txt", image_nameP, sd);

				}

				//畳み込みを行うか既に畳み込んであるデータを用いるかどうか
				switch (no_convolution_frag) {

				case 1://データを用いる
					sprintf(date_buf, no_date);
					std::tie(*Inputimage, image_x, image_y, image_xt, image_yt) = read_log(paramerter, paramerter_count, sd, date_buf, Inputimage);
					printf("inputimage_main=%s\n", Inputimage);
					//convolution_maching(simulation_pattern, image_x, image_y, image_xt, image_yt, paramerter, paramerter_count, sd, date, date_buf, date_directory, threshold, threshold_low, Inputimage);
					if (simulation_pattern[8] != 0) {
						image_xt = image_xt - (2 * simulation_pattern[8]);
						image_yt = image_yt - (2 * simulation_pattern[8]);
					}
					std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = convolution_maching(simulation_pattern, image_x, image_y, image_xt, image_yt, paramerter, paramerter_count, sd, date_buf, date, date_directory, threshold, threshold_low, Inputimage);
					break;
				default://畳み込む
					convolution(argc, argv, image_nameP2, image_x, image_y, image_xt, image_yt, paramerter, paramerter_count, sd, date, date_directory, Inputimage);
					printf("inputiamge=%s\n", Inputimage);
					if (simulation_pattern[8] != 0) {
						image_xt = image_xt - (2 * simulation_pattern[8]);
						image_yt = image_yt - (2 * simulation_pattern[8]);
					}
					//	int time_a1 = msectime();
					std::tie(max_x, max_y, count_tied_V_vote, V_vote_max) = convolution_maching(simulation_pattern, image_x, image_y, image_xt, image_yt, paramerter, paramerter_count, sd, date, date, date_directory, threshold, threshold_low, Inputimage);
					//	int time_a2 = msectime();
					//	printf("t1=%d,t2=%d", time_a1, time_a2);
					//	return 0;
					break;


				}

				//			}
							//}

							//データの統計処理を行う
				if (data_num != 0) {
					int correct_score = 0;
					int correct_x;
					int correct_y;
					correct_score = score(correct_x, correct_y, image_xt, image_yt, max_x, max_y, count_tied_V_vote, V_vote_max, data_num, correct_scv, high_score_range_x, high_score_range_y);
					score_record(date_directory, data_num, Inputimage, correct_score, count_tied_V_vote, V_vote_max, max_x, max_y);
				}
			}

		}
		//	exit(0);
	}//ここを消した
//}	//sd
		printf("全ての処理が終了しました\n");

		return 0;
}




