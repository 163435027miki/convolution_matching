#include <opencv2/opencv.hpp>	//画像読み込み


int write_frame(char date_directory[], char Inputiamge[], std::vector<int> max_x, std::vector<int> max_y, int image_xt,int image_yt,int count_tied_V_vote,int V_vote_max){

	char result_image_name[128];
	printf("inputiamge=%s\n", Inputiamge);
	cv::Mat ImputImageR = cv::imread(Inputiamge);	//入力画像の読み込み．opencv
	for (int i = 0; i < count_tied_V_vote; ++i) {

		cv::rectangle(ImputImageR, cv::Point(max_x[i] - 1, max_y[i] - 1), cv::Point(max_x[i] + image_xt + 1, max_y[i] + image_yt + 1), cv::Scalar(0, 0, 200), 1, 0);

	}

	//画面表示
	std::string windowName = "frame_image";
	cv::namedWindow(windowName);
	cv::imshow(windowName, ImputImageR);

	//画像保存
	sprintf(result_image_name,"%sresult_image_%d.bmp", date_directory, V_vote_max);
	imwrite(result_image_name, ImputImageR);

	cv::waitKey(20);

	return 0;
}