/////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////�֐��̊K�w/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
	--main
		|--timeset																	//���Ԃ̎擾
		|--or--convolution															//��ݍ��݂��s��
		|		|--read_property													//��ݍ��݂�property��ǂݎ��
		|		|--set_outputfile													//��ݍ��݌��ʂ�ۑ�����t�@�C���E�t�H���_�̐ݒ�
		|		|--read_filter														//��ݍ��݂ɗp����t�B���^��ǂݍ���
		|		|--or--convolution_calculation										//�T���Ώۉ摜�ŃJ�[�l����p������ݍ��݂��s���ꍇ
		|		|  or--convolution_gaus_sobel										//�T���Ώۉ摜��sobel�t�B���^����gaus�t�B���^��p����ꍇ
		|		|		|--read_filter_gaus											//�T���Ώۉ摜�ŏ�ݍ��݂ɗp����t�B���^�Ɠ����̃T�C�Y��gaus�t�B���^��ǂݍ���
		|		|--write_file														//�T���Ώۉ摜�ŏ�ݍ��݌��ʂ�csv�t�@�C���o��			
		|		|
		|		|--or--convolution_calculation										//template�摜�ŃJ�[�l����p������ݍ��݂��s���ꍇ
		|		|  or--convolution_gaus_sobel										//template�摜��sobel�t�B���^����gaus�t�B���^��p����ꍇ
		|		|		|--read_filter_gaus											//template�摜�ŏ�ݍ��݂ɗp����t�B���^�Ɠ����̃T�C�Y��gaus�t�B���^��ǂݍ���
		|		|--write_file														//template�摜�ŏ�ݍ��݌��ʂ�csv�t�@�C���o�H
		|		|
		|		|--make_bmp				�~2�~8										//��ݍ��݉摜�̍쐬
		|
		|	--or--read_log															//��ݍ��܂��ɁC���łɏ�ݍ��݂��܂ꂽ�f�[�^��p����													
		|
		|--convolution_maching														//��ݍ��݉摜��p�����u���b�N�}�b�`���O
		|	|--set_convolutionfile													//��ݍ��݃t�@�C���̓ǂݍ���	
		|	|--output_file															//output�t�@�C���̐ݒ�
		|	|--threshold_data_edit													//threshold�ɗp����z�񂪓Ɠ��̏ꍇ�͂����Őݒ肷��
		|	|--edge_st_temp															//���ʕ��͖@�i��Â�2�l���j��臒l�����߂�	
		|	|	|--discriminantAnalysis												//���ʕ��͖@�i��Â�2�l���j��臒l�����߂�
		|	|		|--hist_hozon													//���ʕ��͖@��臒l�����߂�ۂɗp�����q�X�g�O������ۑ�����
		|	|
		|	|--chika_bmp															//2�l���摜�̍쐬
		|	|--or--Sum8_maching														//CB��8�������v����D�܂�C�ő哊�[�����̓u���b�N���Ɠ�����
		|	|		|--Voting_rights_template_sum8									//�e���v���[�g�ɂ��ē��[�����m�F����
		|	|		|--Voting_rights_sum8											//�T���Ώۉ摜�ɂ��ē��[�����m�F����
		|	|		|--vote_maching_sum8											//�}�b�`���O���s��
		|	|			|--max_v_vote_calculate										//���[���̍ő�l�����߂�
		|	|
		|	|  or--Each8_maching													//CB��8�������ꂼ��ŋ��߂�D�܂�C�ő哊�[�����̓u���b�N���~8�D�������C��ݍ��݌���1�݂̂�p����ꍇ������
		|	|		|--Voting_rights_template_each8									//�e���v���[�g�ɂ��ē��[�����m�F����
		|	|		|--Voting_rights_each8											//�T���Ώۉ摜�ɂ��ē��[�����m�F����
		|	|		|--vote_maching_each8											//�}�b�`���O���s��
		|	|			|--max_v_vote_calculate										//���[���̍ő�l�����߂�
		|	|
		|	|	or--Select8_maching													//�e���W�ɂ���V�̃Q�C�����r���C�ł��傫�Ȓl��p����CB�����߂�D�܂�C�ő哊�[�����̓u���b�N���Ɠ������D�ȉ���Each8_maching�Ɠ���
		|	|		|--Voting_rights_template_each8									//�e���v���[�g�ɂ��ē��[�����m�F����
		|	|		|--Voting_rights_each8											//�T���Ώۉ摜�ɂ��ē��[�����m�F����
		|	|		|--vote_maching_each8											//�}�b�`���O���s��
		|	|			|--max_v_vote_calculate										//���[���̍ő�l�����߂�
		|	|
		|	|--write_frame															//�T���Ώۉ摜�Ƀ}�b�`���O���ʂ̃t���[�����L������
		|
		|--score																	//���v�������s�����߁C���_���v�Z����
		|	|--read_correct_xy														//���v�����̂��߁C�������ʒu�̃f�[�^�Z�b�g��ǂݍ���
		|
		|--score_record																//���_���L�^����

*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include<fstream>
#include<iostream>
#include<string>
#include<sstream>	//�����X�g���[��
#include <direct.h>	//�t�H���_���쐬����
#include<thread>	//�����X���b�h
#include<vector>

using namespace std;

int image_x, image_y;		//�摜�T�C�Y
int image_xt, image_yt;		//�摜�T�C�Y

char date[128] = "";
char date_buf[128] = "";
//�o�̓t�@�C���f�B���N�g��
char date_directory[128];
char image_nameP[128];
char image_nameP2[256];
int sd;

char Inputimage[128];



//kernel�̃p�����[�^��sobel�̃T�C�Y���L��
int paramerter_kernel[4] = { 1,3,100,100 };
int paramerter_sobel[4] = { 0,3,5,7 };

int msectime();//msec�P�ʂ̎��Ԏ擾

int timeset(char date[]);
//int notimeset(char date[], int pixel[], int Togire[], int z2, int z);
int convolution(int argc, char** argv, char image_nameP2[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char date[], char date_directory[], char Inputimage[]);

std::tuple<std::vector<int>, std::vector<int>, int, int>convolution_maching(int simulation_pattern[], int &image_x, int &image_y, int &image_xt, int &image_yt, int paramerter[], int paramerter_count, int sd, char conv_date[], char output_date[], char date_directory[], double threshold, double threshold_low, char Inputiamge[]);
//int convolution_maching(int simulation_pattern[],int &image_x, int &image_y,  int &image_xt, int &image_yt,int paramerter[], int paramerter_count, int sd, char conv_date[],char output_date[], char date_directory[], double threshold,  double threshold_low, char Inputiamge[]);

std::tuple<char&, int, int, int, int> read_log(int paramerter[], int paramerter_count, int sd, char date[], char Inputiamge[]);
int score(int correct_x, int correct_y, int image_xt, int image_yt, std::vector<int> max_x, std::vector<int> max_y, int count_tied_V_vote, int V_vote_max, int data_num, char correct_scv[], int high_score_range_x, int high_score_range_y);
int score_record(char date_directory[], int data_num,char Inputimage[], int correct_score, int count_tied_V_vote, int V_vote_max, std::vector<int> max_x, std::vector<int> max_y);



int main(int argc, char** argv) {

	//�ŏI�I�ɋ��߂�������
	std::vector<int>max_x;
	std::vector<int>max_y;
	int count_tied_V_vote;
	int V_vote_max;


	//int pixel[10]={0,1,3,5,7,9,13,17};
	//int Togire[10] = { 0,1,3,5,7,9,13,17 };

	int paramerter[4];					//paramerter[0]=1��sobel�t�B���^,paramerter[0]=2��gaus�~sobel�t�B���^
	paramerter[0] = 0;					//paramerter[0]=1��sobel�t�B���^,paramerter[0]=2��gaus�~sobel�t�B���^

	double threshold = 1;
	double threshold_low = threshold*-1;
	//double threshold_low = -3;



	int simulation_pattern[9] = { 8, 0,0,0,9,011110000,180,5,20 };
	int sd_buf = 0;
	//��@����ʂ��邽�߂�simulation_pattern[0]��ύX����
	/*
							|		0			|		1			|		2		|		3		|		4				|		5			|
	�u���b�N�T�C�YBs[0]		|		8																|
	��ݍ��ނ��ǂ����F[1]	|	��ݍ��݂���	|	��ݍ��݂Ȃ�	|
	�p����2�l����	:[2]	|	2�l���Ȃ�		|					|	2�l��		|	3�l��		|
	threshold��p����:[3]	|		�Ȃ�		|		����		|���臒l�̂�	|����臒l�̂�	|��Â�2�l��,�G�b�W���x|��Â�2�l���I��3�l��|
	���[���@		:[4]	|	---																	|8:�u���b�N���~8		|	9:�u���b�N��	|10:�����d���ɂ���ėp���镔����ύX����|
	�p�������		:[5]	|���������p����Ȃ�1|1~8:�p�������
	���v�������s��	:[6]	|	�s��Ȃ�		|�s���C���f�[�^�̐�|
	�g���܂Ƃ߂�	:[7]	|�܂Ƃ߂Ȃ�			|	���e����Y��������
	���[�����D		:[8]	|���ォ�炱�̒l�������[���𔍒D����D��ݍ��݂͑S��f�ɍs��(��F�e���v���[�g�摜120�~120��20�ɐݒ聨80�~80��pixel�̏�ݍ��݉摜�𓊕[�ɗp����)
	*/


	//�W���΍��̒����ӏ�
	int sd_max = 50;
	int paramerter_count_max = 1;


	//�������W�̌덷
	int high_score_range_x = 0;
	int high_score_range_y = 0;
	high_score_range_x = simulation_pattern[7];
	high_score_range_y = simulation_pattern[7];

	//��ݍ��݂̐���
	int no_convolution_frag = simulation_pattern[1];		//1���Ə�ݍ��݂��s��Ȃ��D
	int data_num = simulation_pattern[6];					//�f�[�^��ǂݎ��Ƃ��C�f�[�^�̐�
	char no_date[128];										//�o�̓t�H���_�������ł͂Ȃ��ʂɐݒ肵�����Ƃ��D�p����f�[�^�̊K�w�ɐ��������邽�ߒ���
	//char no_date_directory_buf[128] = "simulation18-0115\\";//�o�̓t�H���_�������ł͂Ȃ��ʂɐݒ肵�����Ƃ��D�p����f�[�^�̊K�w�ɐ��������邽�ߒ���
	char no_date_directory_buf[128];
	//char *no_date_directory_buf = "simulation18-0115_sobel\\";//�o�̓t�H���_�������ł͂Ȃ��ʂɐݒ肵�����Ƃ��D�p����f�[�^�̊K�w�ɐ��������邽�ߒ���
	//sprintf(no_date, "convolution\\gpu2");

	//
	char *correct_scv = "..\\bmp\\simulation18-0115\\correct.csv";		//�p����T���������W

	int paramerter_count = 0;									//�p����p�����[�^�̔ԍ�

	//�p����p�����[�^����
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
		printf("paramerter[0]�̒l����������\nparamerter[0]=%d\n", paramerter[0]);
		return 0;
	}



//		for (sd = 0; sd <= sd_max; sd = sd + 10) {
			sprintf(no_date_directory_buf, "simulation18-0115_sd%d\\", sd);
	for (int image_kurikaeshi = 1; image_kurikaeshi < data_num + 1; ++image_kurikaeshi) {
		//	int image_kurikaeshi = 1;
		
		if (no_convolution_frag == 1)sprintf(no_date, "%s%d", no_date_directory_buf, image_kurikaeshi);
		//if (no_convolution_frag == 1)sprintf(no_date, "%s93", no_date_directory_buf);

		timeset(date);									//���Ԃ̎擾
//		sprintf(date, "sd%d\\%d", sd, image_kurikaeshi);			//�o�̓t�H���_�������ł͂Ȃ��ʂɐݒ肵�����Ƃ��D�p����f�[�^�̊K�w�ɐ��������邽�ߒ���
		sprintf(date, "%d", image_kurikaeshi);			//�o�̓t�H���_�������ł͂Ȃ��ʂɐݒ肵�����Ƃ��D�p����f�[�^�̊K�w�ɐ��������邽�ߒ���
		//sprintf(date, "93");

		for (int paramerter_count = 1; paramerter_count <= paramerter_count_max; ++paramerter_count) {
			//	for (int sim_num = 1; sim_num < 4; ++sim_num) {

			for (sd = sd_buf; sd <= sd_buf; sd = sd + 10) {
				//			for (sd = 10; sd <= sd_max; sd = sd + 10) {

								//sobel�t�B���^��p���鎞�̐ݒ�
				if (paramerter[0] == 1 || paramerter[0] == 2) {

					sprintf(image_nameP, "..\\property_usa\\simulation18-0115\\property_%d�~%dsobel_conv_", paramerter[paramerter_count], paramerter[paramerter_count]);
					sprintf(image_nameP2, "%ssd%d_%d.txt", image_nameP, sd, image_kurikaeshi);

					//kernel��p���鎞�̐ݒ�
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

				//��ݍ��݂��s�������ɏ�ݍ���ł���f�[�^��p���邩�ǂ���
				switch (no_convolution_frag) {

				case 1://�f�[�^��p����
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
				default://��ݍ���
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

							//�f�[�^�̓��v�������s��
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
	}//������������
//}	//sd
		printf("�S�Ă̏������I�����܂���\n");

		return 0;
}




