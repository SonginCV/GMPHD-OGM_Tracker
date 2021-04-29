/*
BSD 2-Clause License

Copyright (c) 2019, Young-min Song,
Machine Learning and Vision Lab (https://sites.google.com/view/mlv/),
Gwangju Institute of Science and Technology(GIST), South Korea.
All rights reserved.

This software is an implementation of the GMPHD-OGM tracker,
which not only refers to the paper entitled
"Online Multi-Object Tracking with GMPHD Filter and Occlusion Group Management"
but also is available at https://github.com/SonginCV/GMPHD-OGM.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// demo_GMPHD-OGM.cpp
#pragma once

#include "stdafx.h"

using namespace std;

#define DB_TYPE_MOT15	0	// MOT Challenge 2015 Dataset
#define DB_TYPE_MOT17	1	// MOT Challenge 2017 Dataset
#define DB_TYPE_CVPR19	2   // MOT Challenge 2019 (CVPR 2019) Dataset

namespace SYM {
	enum {
		OBJ_TYPE_PERSON = 0, OBJ_TYPE_CAR = 1,
		OBJ_TYPE_BICYCLE = 2, OBJ_TYPE_TRUCK = 3,
		OBJ_TYPE_BUS = 4, OBJ_TYPE_DONT_CARE = 5
	};
	std::string OBJECT_STRINGS[6] = \
	{ "Person", "Car",
		"Bicycle", "Truck",
		"Bus", "DontCare"};

	std::string DB_NAMES[3] = { "MOT15", "MOT17", "CVPR19" };

	CvScalar OBJECT_COLORS[7] = \
	{	cvScalar(255, 255, 0),	/*Person: Lite Blue*/
		cvScalar(255, 255, 255),/*Car: White*/
		cvScalar(255, 0, 255),	/*Bicycle: Pink*/
		cvScalar(0, 0, 255),	/*Truck: Red*/
		cvScalar(0, 255, 255),	/*Bus: Yellow*/
		cvScalar(0, 255, 0),	/*Green*/
		cvScalar(255, 0, 0)		/*Blue*/};
}

/**
* @brief		The Function for Loading the Experimental Environments (including Datasets, Parameters,...)
* @details
* @param[in]	seqList		The file path of sequences' list txt	(e.g., seqs\\MOT15train.txt).
* @param[in]	paramsFile	The file path of parameter txt			(e.g., params\\MOT15train.txt).
* @param[out]	seqNames	The list of all sequneces' names from seqList file.
* @param[out]	seqPaths	The folder path of all image sequences.
* @param[out]	allDets		The detection results for the image sequences.
* @param[out]	sceneParams The scene-parameters from paramsFile txt.
* @return
* @throws
*/
void LoadEnvSettings(const string seqFile, const string paramsFile, vector<string>& seqNames, vector<string>& seqPaths, vector<vector<vector<vector<float>>>>& allDets, vector<GMPHDOGMparams>& sceneParams);
// Read All Images' Path and Detections as strings 
void ReadAllSequences(const string seqFile, vector<string>& seqNames, vector<string>& seqPaths, vector<string>& detTxts);
// The Functions for Reading Scene Parameters from txt
vector<GMPHDOGMparams> ReadSceneOptParams(int nums_of_scenes, string paramsFilePath, int MERGE_METRIC);
void TokenizeParamStrings(vector<string> src, vector<GMPHDOGMparams>& dst);
// The Function for Reading Detection Results for txt
void ReadAllDetections(const vector<string> detTxts, vector<vector<vector<vector<float>>>>& allSeqDets);
// The Function for Sorting Detection Responses by frame number (ascending order)
vector<string> SortAllDetections(const vector<string> allLines, int DB_TYPE = DB_TYPE_MOT17);

/**
* @brief		The Function for Testing the GMPHD-OGM tracker in All Scenes
* @details
* @param[in]	seqNames		The sequences' names.
* @param[in]	imgRootPaths	The folder paths for the image sequences.
* @param[in]	allSeqDets		THe detection results' for the image sequences. 
* @param[in]	sceneParams		The parameters for the GMPHD-OGM tracker on all the scenes.
* @param[out]	totalProcSecs	The processing time (seconds) of the GMPHD-OGM tracker on all the scenes.
* @return		int				The total number of all the image sequences.
* @throws
*/
int DoBatchTest(const vector<string> seqNames, const vector<string> imgRootPaths, const vector<vector<vector<vector<float>>>> allSeqDets, const vector<GMPHDOGMparams> sceneParams, double& totalProcSecs);
/**
* @brief		The Function for Testing the GMPHD-OGM tracker in One Scene
* @details
* @param[in]	seqName			The name of a sequence.
* @param[in]	imgFolderPath	The folder path of the image sequence.
* @param[in]	seqDets			The detection results on the sequence.
* @param[in]	params			The parameters for the GMPHD-OGM tracker on the scene.
* @param[out]	procSecs		The processing time (seconds) of the GMPHD-OGM tracker on the scene.
* @return		int				The number of the image sequence.
* @throws
*/
int DoSequenceTest(const string seqName, const string imgFolderPath, const vector<vector<vector<float>>> seqDets, const GMPHDOGMparams params, double& procSecs);
// The Function for Approximating the Lost Tracks which were recovered by T2TA.
void InterpolateAllTracks(vector<vector<BBTrk>> inputTracks, vector<vector<BBTrk>>& outputTracks);
/**
* @brief	The Function for Writing the Tracking Results into a Text File
* @details
* @param
* @param
* @param
* @return
* @throws
*/
void WriteTracksTxt(const int DB_TYPE, string train_or_test, string seqName, GMPHD_OGM* tracker);
void CreateDirsRecursive(string path);
// Drawing Functions
// Draw Detection and Tracking Results
void DrawDetandTrk(cv::Mat& img_det, cv::Mat& img_trk, GMPHD_OGM tracker, const vector<vector<float>> dets, const vector<vector<float>> trks);
/// Draw the Detection Bounding Box
void DrawDetBBS(cv::Mat& img, int iter, cv::Rect bb, double conf, double conf_th, int digits, cv::Scalar color, int thick = 3);
/// Draw the Tracking Bounding Box
void DrawTrkBBS(cv::Mat& img, cv::Rect rec, cv::Scalar color, int thick, int id, double fontScale, string type);
// Draw Frame Number and FPS
void DrawFrameNumberAndFPS(int iFrameCnt, cv::Mat& img, double scale, int thick, int frameOffset = 0, int frames_skip_interval = 1, double sec = -1.0);

// Global Environmental Settings
int DB_TYPE = DB_TYPE_MOT17;	// DB_TYPE_MOT15 or DB_TYPE_MOT17
string mode = "train";			// "train" or "test"
string detector = "FRCNN";		// "ACF", "DPM", "FRCNN", or "SDP"

int main()
{
	// Load Local Environmental Settings including image sequences, detection results, and the parameters for the GMPHD-OGM tracker
	/// MOT15 or MOT17
	// MOT15train.txt, MOT17train_DPM.txt, MOT17train_FRCNN.txt, or MOT17train_SDP.txt 
	string seqList = "seqs\\" + SYM::DB_NAMES[DB_TYPE] + mode + "_" + detector + ".txt";
	string paramsFile = "params\\" + SYM::DB_NAMES[DB_TYPE] + mode + "_" + detector + ".txt";

	vector<string> seqNames,imgRootPaths;
	vector<vector<vector<vector<float>>>> allDets;
	vector<GMPHDOGMparams> sceneParams;

	LoadEnvSettings(seqList, paramsFile, seqNames, imgRootPaths, allDets, sceneParams);

	// Test "One Image Sequences" or "Batch: More than One Image Sequences"
	int64 t_start= cv::getTickCount();
	double procSecsDoMOT = 0.0;
	//int totalProcFrames = DoSequenceTest(seqNames[2], imgRootPaths[2], allDets[2], sceneParams[2], procSecsDoMOT);
	int totalProcFrames = DoBatchTest(seqNames, imgRootPaths, allDets, sceneParams, procSecsDoMOT);
	int64 t_end = cv::getTickCount();
	float totalProcSecs = (float)((t_end - t_start) / cv::getTickFrequency());
	
	cout << "Total processing frames: " << totalProcFrames << "." << endl;
	cout << "Total processing time: " << totalProcSecs << " secs. (DoMOT: " << procSecsDoMOT <<" secs.)"<< endl;
	cout << "Avg. frames per second: " << totalProcFrames / totalProcSecs << " FPS. (DoMOT: " << totalProcFrames/procSecsDoMOT <<" FPS)" << endl;

	return 0;
}


// The Function for Loading the Experimental Environments (including Datasets, Parameters,...)
void LoadEnvSettings(const string seqList, const string paramsFile, vector<string>& seqNames, vector<string>& seqPaths, vector<vector<vector<vector<float>>>>& allDets, vector<GMPHDOGMparams>& sceneParams) {
	cout << "Images and detections are loaded: \"" << seqList << "\"" << endl;
	vector<string> detTxts;
	ReadAllSequences(seqList, seqNames, seqPaths, detTxts);
	ReadAllDetections(detTxts, allDets);

	cout << "Scene parameters are loaded: \"" << paramsFile << "\"" << endl;
	sceneParams = ReadSceneOptParams(seqPaths.size(), paramsFile, MERGE_METRIC_OPT);
}
void ReadAllSequences(const string seqFile, vector<string>& seqNames, vector<string>& seqPaths, vector<string>& detTxts) {

	vector<string> allLines;
	if (_access(seqFile.c_str(), 0) == 0) {
		ifstream infile(seqFile);

		string line,dataFolder;
		if (getline(infile, line)) {
			dataFolder = line;// The first line indicates dataset's root location.
		}
		cout << line << endl;
		int sq = 1;
		while (getline(infile, line)) {
			string imgPath = dataFolder + line +"\\img1\\";
			string detPath = dataFolder + line +"\\det\\det.txt";
			cout << sq++ <<": "<< line << endl;
			seqNames.push_back(line);
			if (_access(imgPath.c_str(), 0) == 0)	seqPaths.push_back(imgPath);
			else									cout << imgPath << " doesn't exist!!" << endl;
			if (_access(detPath.c_str(), 0) == 0)	detTxts.push_back(detPath);
			else									cout << detPath << " doesn't exist!!" << endl;
		}
		//cout << endl;
	}
	else {
		printf("%s doesn't exist!\n", seqFile.c_str());
	}
}
vector<GMPHDOGMparams> ReadSceneOptParams(int nums_of_scenes, string paramsFilePath, int MERGE_METRIC) {
	vector<string> str_params_IOU, str_params_SIOA;
	ifstream infile(paramsFilePath);
	string line;
	int cnt = 1;
	while (getline(infile, line)) {
		if (cnt > 1) {
			if (cnt >= 3 && cnt < 3 + nums_of_scenes) {
				str_params_SIOA.push_back(line);
				//cout << line << endl;
			}
			else if (cnt >= 4 + nums_of_scenes && cnt < 4 + 2 * nums_of_scenes) {
				str_params_IOU.push_back(line);
				//cout << line << endl;
			}
		}
		++cnt;
	}

	vector<GMPHDOGMparams> params_IOU, params_SIOA;

	TokenizeParamStrings(str_params_SIOA, params_SIOA);
	TokenizeParamStrings(str_params_IOU, params_IOU);

	if (MERGE_METRIC == MERGE_METRIC_IOU)		return params_IOU;
	else if (MERGE_METRIC == MERGE_METRIC_SIOA)	return params_SIOA;
}
void TokenizeParamStrings(vector<string> src, vector<GMPHDOGMparams>& dst) {

	vector<string>::iterator iter;
	for (iter = src.begin(); iter != src.end(); ++iter) {
		boost::char_separator<char> bTok(", ");
		boost::tokenizer < boost::char_separator<char>>tokens(iter[0], bTok);
		vector<string> vals;
		for (const auto& t : tokens)
		{
			vals.push_back(t);
		}

		GMPHDOGMparams tParams;
		tParams.DET_MIN_CONF = boost::lexical_cast<double>(vals.at(1));		// Detection Confidence Threshold
		tParams.T2TA_MAX_INTERVAL = boost::lexical_cast<int>(vals.at(2));	// T2TA Maximum Interval
		tParams.TRACK_MIN_SIZE = boost::lexical_cast<int>(vals.at(3));		// Track Minium Length
		tParams.FRAMES_DELAY_SIZE = tParams.TRACK_MIN_SIZE - 1;
		tParams.GROUP_QUEUE_SIZE = tParams.TRACK_MIN_SIZE * 10;

		//cout << tParams.DET_MIN_CONF << ", " << tParams.T2TA_MAX_INTERVAL << "," << tParams.TRACK_MIN_SIZE << endl;

		if (vals.size() > 4)		boost::lexical_cast<int>(vals.at(4));	// Optional Settings (IOU or L1L1 -> D2TA_T2TA : L1 or L2 norm)

		dst.push_back(tParams);
	}
}
vector<string> SortAllDetections(const vector<string> allLines, int DB_TYPE) {
	// ascending sort by frame number
	/// http://azza.tistory.com/entry/STL-vector-%EC%9D%98-%EC%A0%95%EB%A0%AC

	class T {
	public:
		int frameNum;
		string line;
		T(string s, int DB_TYPE) {
			line = s;

			char tok[8];
			if (DB_TYPE == DB_TYPE_MOT15 || DB_TYPE == DB_TYPE_MOT17 || DB_TYPE == DB_TYPE_CVPR19) {
				strcpy_s(tok, ", ");
			}

			boost::char_separator<char> bTok(tok);

			boost::tokenizer < boost::char_separator<char>>tokens(s, bTok);
			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			frameNum = boost::lexical_cast<int>(vals.at(0));
		}
		bool operator<(const T &t) const {
			return (frameNum < t.frameNum);
		}
	};


	// Reconstruct the vector<T> from vector<string> for sorting
	vector<T> tempAllLines;
	vector<string>::const_iterator iter = allLines.begin();
	for (; iter != allLines.end(); iter++) {
		if (iter[0].size() < 2) continue;

		tempAllLines.push_back(T(iter[0], DB_TYPE));
	}
	// Sort the vector<T> by frame number
	std::sort(tempAllLines.begin(), tempAllLines.end());

	// Copy the sorted vector<T> to vector<string>
	vector<string> sortedAllLines;
	vector<T>::iterator iterT = tempAllLines.begin();
	for (; iterT != tempAllLines.end(); iterT++) {

		sortedAllLines.push_back(iterT[0].line);
	}

	return sortedAllLines;
}
void ReadAllDetections(const vector<string> detTxts, vector<vector<vector<vector<float>>>>& allDets) {

	// Read All Detections with Strings
	vector<string>::const_iterator it;
	vector<vector<string>> allDetLines; // all detections of all sequences (in forms of strings)
	//int i = 0;
	for (it = detTxts.begin(); it != detTxts.end(); ++it) {
		vector<string> detLines; 
		if (_access(it[0].c_str(), 0) == 0) {
			ifstream infile(it[0]);
			string line;

			while (!infile.eof()) {
				getline(infile, line);
				detLines.push_back(line);
				//cout << line << endl;
			}
		}
		else {
			printf("%s doesn't exist!\n", it[0].c_str());
		}
		detLines = SortAllDetections(detLines, DB_TYPE);
		allDetLines.push_back(detLines);
	}

	// Convert Strings into vector<float>
	vector<vector<string>>::iterator itSeqs;
	for (itSeqs = allDetLines.begin(); itSeqs != allDetLines.end(); ++itSeqs) {

		vector<vector<vector<float>>> bbsSeq;
		vector<vector<float>> bbsFrame;
		vector<string>::iterator itLines;
		int iFrmCnt = 1;
		int cnt = 1;
		for (itLines = itSeqs[0].begin(); itLines != itSeqs[0].end(); ++itLines) {

			boost::char_separator<char> bTok(", ");
			boost::tokenizer < boost::char_separator<char>>tokens(itLines[0], bTok);

			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			if (vals.empty()) {
				bbsSeq.push_back(bbsFrame); // is deep copy?
				bbsFrame.clear();
				break;
			}
			int curFrm;
			curFrm = (int)boost::lexical_cast<float>(vals.at(0));		// frame number

			vector<float> bb;
			bb.push_back(curFrm);									// frame number				
			bb.push_back(boost::lexical_cast<float>(vals.at(2)));	// y
			bb.push_back(boost::lexical_cast<float>(vals.at(3)));	// y
			bb.push_back(boost::lexical_cast<float>(vals.at(4)));	// width
			bb.push_back(boost::lexical_cast<float>(vals.at(5)));	// height
			bb.push_back(boost::lexical_cast<float>(vals.at(6)));	// detection score
			if (iFrmCnt == curFrm) {
				bbsFrame.push_back(bb);
				//printf("%d:(%d:%d,%d,%d,%d,%.2f)\n", (int)bb[0], bbsFrame.size(),(int)bb[1], (int)bb[2], (int)bb[3], (int)bb[4], bb[5]);
			}
			else if (iFrmCnt < curFrm) { // Next frame
				bbsSeq.push_back(bbsFrame); // is deep copy? Yes
				bbsFrame.clear();
				bbsFrame.push_back(bb);
				iFrmCnt++;
			}
		}
		// End Frame
		bbsSeq.push_back(bbsFrame);
		allDets.push_back(bbsSeq);
	}
}
//	The Function for Testing the GMPHD-OGM tracker in All Scenes
int DoBatchTest(const vector<string> seqNames, const vector<string> imgRootPaths, const vector<vector<vector<vector<float>>>> allSeqDets, const vector<GMPHDOGMparams> sceneParams, double& totalProcSecs) {

	int totalFrames = 0;
	totalProcSecs = 0.0;
	for (int sq = 0; sq<imgRootPaths.size(); ++sq) {
		cout << "-Sequence " << sq + 1 << ": ";
		double procSecs = 0.0;
		totalFrames += DoSequenceTest(seqNames[sq], imgRootPaths[sq], allSeqDets[sq], sceneParams[sq], procSecs);
		totalProcSecs += procSecs;
	}
	return totalFrames;
}
// The Function for Testing the GMPHD-OGM tracker in One Scene
int DoSequenceTest(const string seqName, const string imgFolderPath, const vector<vector<vector<float>>> seqDets, const GMPHDOGMparams params, double& procSecs) {
	// Init a Tracker
	GMPHD_OGM *tracker=new GMPHD_OGM();
	tracker->SetParams(params);

	// vector<string> for All Images' Paths
	vector<string> imgs;

	// Read Image Files Paths
	boost::filesystem::path p(imgFolderPath);

	boost::filesystem::directory_iterator end_itr;
	// cycle through the directory
	for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr)
	{
		// If it's not a directory, list it. If you want to list directories too, just remove this check.
		if (boost::filesystem::is_regular_file(itr->path())) {
			// assign current file name to current_file and echo it out to the console.
			string imgFile = itr->path().string();
			imgs.push_back(imgFile);
		}
	}
	// Specification of the Images and Detections
	cv::Mat tImg = cv::imread(imgs[0]);
	int frmWidth = tImg.cols, frmHeight = tImg.rows;
	tImg.release();
	int nImages = imgs.size();
	tracker->SetTotalFrames(nImages);
	int sumDets = 0;
	for (int iFrmCnt = 0; iFrmCnt < nImages; ++iFrmCnt) {
		sumDets += seqDets[iFrmCnt].size();
	}
	printf("Tracking in %d (%dx%d) images with (total detections:%d, density:%.2lf)\n",nImages, frmWidth, frmHeight, sumDets, sumDets/(float)nImages);
	if(tracker->GetParams().DET_MIN_CONF == -100.00)	printf("  (ALL, %d, %d)\n", tracker->GetParams().T2TA_MAX_INTERVAL, tracker->GetParams().TRACK_MIN_SIZE);
	else												printf("  (%.2lf, %d, %d)\n", tracker->GetParams().DET_MIN_CONF, tracker->GetParams().T2TA_MAX_INTERVAL, tracker->GetParams().TRACK_MIN_SIZE);
	if (VISUALIZATION_MAIN_ON) {
		cv::namedWindow("Detection");	cv::moveWindow("Detection", 0, 0);
		cv::namedWindow("Tracking");	cv::moveWindow("Tracking", frmWidth + 10, 0);
	}

	// Multi-Object Tracking
	// Tracking-by-Detection Paradigm: input is detections
	// Online Approach: load detections and do tracking, frame-by-frame
	procSecs = 0.0;
	for (int iFrmCnt = 0; iFrmCnt<nImages; ++iFrmCnt) {

		// Image
		cv::Mat img = cv::imread(imgs[iFrmCnt]);
		cv::Mat img_det = img.clone();
		cv::Mat img_trk = img.clone();

		// Tracking
		double t_start = (double)cv::getTickCount();
		vector<vector<float>> tracks = tracker->DoMOT(iFrmCnt, img_trk, seqDets[iFrmCnt]);
		double t_end = (double)cv::getTickCount();
		double sec = (t_end- t_start)/cv::getTickFrequency();
		procSecs += sec;

		// Visualization
		/// Console
		std::cerr << "(" << (iFrmCnt + FRAME_OFFSET);
		std::cerr << "/";
		std::cerr << nImages;
		std::cerr << ") ";
		std::cerr << "\r";
		/// Window
		if (VISUALIZATION_MAIN_ON) {
			cv::Mat img_trk_vis = tracker->imgBatch[0].clone();

			DrawDetandTrk(img_det, img_trk_vis, tracker[0], seqDets[iFrmCnt], tracks);
			DrawFrameNumberAndFPS(iFrmCnt, img_det, 2.0, 2, FRAME_OFFSET);
			DrawFrameNumberAndFPS(iFrmCnt - tracker->GetParams().FRAMES_DELAY_SIZE, img_trk_vis, 2.0, 2, FRAME_OFFSET, 1, sec);
			//DrawFrameNumberAndFPS(iFrmCnt - FRAMES_DELAY+offset, img_trk_delay, 2.0, 2, 0, frames_skip_interval, tTrk);

			cv::imshow("Detection", img_det);
			cv::imshow("Tracking", img_trk_vis);
			if(iFrmCnt==0) cv::waitKey();
			else {
				if(SKIP_FRAME_BY_FRAME) cv::waitKey();
				else					cv::waitKey(10);
			}

			img_trk_vis.release();
		}

		img.release();
		img_det.release();
		img_trk.release();
	}

	// Write the Tracking Results into a txt file.
	WriteTracksTxt(DB_TYPE, mode, seqName, tracker);

	// Free the tracker
	delete tracker;
	
	return nImages;
}
// The Function for Writing the Tracking Results into a Text File
void WriteTracksTxt(const int DB_TYPE, string train_or_test, string seqName, GMPHD_OGM* tracker) {

	vector<vector<BBTrk>> allTracksINTP;
	InterpolateAllTracks(tracker->allLiveReliables, allTracksINTP);

	char filePath[256],dirPath[256];

	if (DB_TYPE == DB_TYPE_MOT15) {

		sprintf_s(dirPath, 256, "res\\MOT15\\%s", train_or_test);
		sprintf_s(filePath, 256, "res\\MOT15\\%s\\%s.txt", train_or_test, seqName);

		CreateDirsRecursive(string(dirPath));
	}
	else if (DB_TYPE == DB_TYPE_MOT17) {
		sprintf_s(dirPath, 256, "res\\MOT17\\%s", train_or_test);
		sprintf_s(filePath, 256, "res\\MOT17\\%s\\%s.txt", train_or_test, seqName);

		CreateDirsRecursive(string(dirPath));
	}

	cout << "   GMPHD-OGM" << ":" << filePath << endl;

	FILE* fp;
	fopen_s(&fp, filePath, "w+");

	for (int i = 0; i < allTracksINTP.size(); ++i) { // frame by frame
		if (!allTracksINTP[i].empty()) {
			for (int tr = 0; tr < allTracksINTP[i].size(); ++tr) {		
				fprintf_s(fp, "%d,%d,%.2lf,%.2f,%.2f,%.2f,-1,-1,-1,-1\n", i + FRAME_OFFSET,\
					allTracksINTP[i][tr].id,\
					(float)allTracksINTP[i][tr].rec.x, (float)allTracksINTP[i][tr].rec.y, (float)allTracksINTP[i][tr].rec.width, (float)allTracksINTP[i][tr].rec.height);
			}
		}
	}
	fclose(fp);
}
void CreateDirsRecursive(string path) {

	boost::char_separator<char> bTok("\\/");
	boost::tokenizer < boost::char_separator<char>>tokens(path, bTok);
	vector<string> vals;
	vector<string> dir_paths;
	string path_appended = ".";
	for (const auto& t : tokens)
	{
		path_appended = path_appended + "\\" + t;
		dir_paths.push_back(path_appended);
	}
	for (const auto& dir : dir_paths) {
		//cout << dir << endl;

		boost::filesystem::path res_folder(dir);
		if (!boost::filesystem::exists(res_folder)) {
			boost::filesystem::create_directory(res_folder);
			//printf("Create %s\n", res_folder.c_str());
		}
	}
}
void InterpolateAllTracks(vector<vector<BBTrk>> inputTracks, vector<vector<BBTrk>>& outputTracks) {
	// Make vector<vector> tracks to map<vector> tracks
	map<int, vector<BBTrk>> allTracks;
	map<int, vector<BBTrk>> allTracksINTP;

	for (int i = 0; i < inputTracks.size(); ++i) {				// iterate all frames
		if (!inputTracks[i].empty()) {
			for (int j = 0; j < inputTracks[i].size(); ++j) {	// iterate objects at a frame


				int id = inputTracks[i][j].id;
				vector<BBTrk> tracklet;
				tracklet.push_back(inputTracks[i][j]);

				pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = allTracks.insert(map<int, vector<BBTrk>>::value_type(id, tracklet));

				if (isEmpty.second == false) { // already has a element with target.at(j).id
					allTracks[id].push_back(inputTracks[i][j]);
				}
			}
		}
	}
	// Find the tracks' lost interval
	map<int, vector<BBTrk>>::iterator iterAllTrk;
	for (iterAllTrk = allTracks.begin(); iterAllTrk != allTracks.end(); ++iterAllTrk) { // iterate all tracks (ID by ID)
		if (!iterAllTrk->second.empty()) {

			if (iterAllTrk->second.size() >= 2) {
				int i = 0;
				do {
					int prev_fn = iterAllTrk->second[i].fn;
					int cur_fn = iterAllTrk->second[i + 1].fn;

					if (prev_fn + 1 < cur_fn) {


						double fd = cur_fn - prev_fn;
						cv::Rect prevRec = iterAllTrk->second[i].rec;
						cv::Rect curRec = iterAllTrk->second[i + 1].rec;
						double xd = (curRec.x - prevRec.x) / fd;
						double yd = (curRec.y - prevRec.y) / fd;
						double wd = (curRec.width - prevRec.width) / fd;
						double hd = (curRec.height - prevRec.height) / fd;

						vector<BBTrk> trks[3]; // 0:track in pre-lost interval, 1: track in lost interval, 2: track in post-lost interval;
						int nSize = iterAllTrk->second.size();
						for (int f = 0; f < nSize; ++f) {
							if (f < i + 1)	trks[0].push_back(iterAllTrk->second[f]);
							else			trks[2].push_back(iterAllTrk->second[f]);
						}

						for (int f = 1; f < fd; ++f) {
							BBTrk intp;//= iterAllTrk->second[i];
							intp.id = iterAllTrk->first;
							intp.fn = prev_fn + f;
							intp.rec.x = prevRec.x + f*xd;
							intp.rec.y = prevRec.y + f*yd;
							intp.rec.width = prevRec.width + f*wd;
							intp.rec.height = prevRec.height + f*hd;
							intp.isInterpolated = true;
							trks[1].push_back(intp);
						}
						vector<BBTrk> intpTrk;
						for (int t = 0; t < 3; t++) {
							for (int f = 0; f < trks[t].size(); f++) {
								intpTrk.push_back(trks[t][f]);
							}
						}
						iterAllTrk->second.clear();
						iterAllTrk->second.assign(intpTrk.begin(), intpTrk.end());
						i = i + fd - 1; // ??
					}

					++i;
					if (i == iterAllTrk->second.size() - 1) break;
				} while (1);
			}

			vector<BBTrk> track = iterAllTrk->second;
			int id = iterAllTrk->first;
			pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = allTracksINTP.insert(map<int, vector<BBTrk>>::value_type(id, track));

			//if (isEmpty.second == false) { // already has a element with target.at(j).id
			//	printf("ID%d is already added\n", iterAllTrk->first);
			//}
		}
	}
	// Make map<vector> tracks to vector<vector> tracks
	map<int, vector<BBTrk>>::iterator iterMapINTP;
	for (int f = 0; f < inputTracks.size(); ++f) {
		map<int, vector<BBTrk>>::iterator iterMapINTP;
		vector<BBTrk> trkVecINTP;
		for (iterMapINTP = allTracksINTP.begin(); iterMapINTP != allTracksINTP.end(); ++iterMapINTP)
		{
			if (!iterMapINTP->second.empty()) {
				vector<BBTrk>::iterator iterT;
				for (iterT = iterMapINTP->second.begin(); iterT != iterMapINTP->second.end(); ++iterT) {
					if (iterT->fn == f) {
						trkVecINTP.push_back(*iterT);
					}
				}

			}
		}
		outputTracks.push_back(trkVecINTP);
	}
}
// Draw Detection and Tracking Results
void DrawDetandTrk(cv::Mat& img_det, cv::Mat& img_trk, GMPHD_OGM tracker, const vector<vector<float>> dets, const vector<vector<float>> trks) {
	vector<vector<float>>::const_iterator iterD;
	int i = 1;
	for (iterD = dets.begin(); iterD != dets.end(); ++iterD) {

		cv::Rect rec((int)iterD[0][1], (int)iterD[0][2], (int)iterD[0][3], (int)iterD[0][4]);
		float confidence = iterD[0][5];

		DrawDetBBS(img_det, i++, rec, confidence, tracker.GetParams().DET_MIN_CONF, 5, SYM::OBJECT_COLORS[0], 4);
	}
	vector<vector<float>>::const_iterator iterT;
	for (iterT = trks.begin(); iterT != trks.end(); ++iterT) {

		int id = iterT[0][0];
		cv::Rect rec((int)iterT[0][1], (int)iterT[0][2], (int)iterT[0][3], (int)iterT[0][4]);

		DrawTrkBBS(img_trk, rec, tracker.color_tab[id % (MAX_OBJECTS - 1)], BOUNDING_BOX_THICK, id, ID_CONFIDENCE_FONT_SIZE - 1, SYM::OBJECT_STRINGS[0]);
	}
}
// Draw the Detection Bounding Box
void DrawDetBBS(cv::Mat& img, int iter, cv::Rect bb, double conf, double conf_th, int digits, cv::Scalar color, int thick) {

	int xc = bb.x + bb.width / 2;
	int yc = bb.y + bb.height / 2;

	std::ostringstream ost;
	ost << conf;
	std::string str = ost.str();
	char cArrConfidence[8];
	int c;
	for (c = 0; c < digits && c < str.size(); c++) cArrConfidence[c] = str.c_str()[c];
	cArrConfidence[c] = '\0';


	if (conf >= conf_th) {
		/// Draw Detection Bounding Boxes with detection cofidence score
		cv::rectangle(img, bb, color, thick);
		cv::rectangle(img, cv::Point(bb.x, bb.y), cv::Point(bb.x + bb.width, bb.y - 30), color, -1);
		cv::putText(img, cArrConfidence, cvPoint(bb.x, bb.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 0, 0), 2);
	}
	else {
		cv::rectangle(img, bb, color, 1);
		//cv::rectangle(img, cv::Point(bb.x, bb.y+ bb.height), cv::Point(bb.x + bb.width, bb.y + bb.height + 30), color, -1);
		//cv::putText(img, cArrConfidence, cvPoint(bb.x, bb.y + bb.height + 15), FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0, 0, 0), 1);
	}

	/// Draw Observation ID (not target ID)
	char cArrObsID[8];
	sprintf_s(cArrObsID, 8, "%d", iter);
	cv::putText(img, cArrObsID, cvPoint(bb.x + 5, bb.y + 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
}
// Draw the Tracking Bounding Box
void DrawTrkBBS(cv::Mat& img, cv::Rect rec, cv::Scalar color, int thick, int id, double fontScale, string type) {

	if (id >= 0) {
		string strID;
		if (type.empty())	strID = to_string(id);
		else			strID = type.substr(0, 1) + " " + to_string(id);
		
		int wid = 0;
		if (id > 0) 	wid = log10f(id);
		else 	 	wid = 0;
		
		int bgRecWidth = fontScale*(int)(wid + 3) * 20;
		int bgRecHeight = fontScale * 40;
		cv::Point pt;
		cv::Rect bg;
		pt.x = rec.x;
		if ((rec.y + rec.height / 2.0) < img.rows / 2) { // y < height/2 (higher)
			pt.y = rec.y + rec.height + 40; // should be located on the bottom of the bouding box
			bg = cv::Rect(rec.x - 5, rec.y + rec.height, bgRecWidth, bgRecHeight + 10);

		}
		else { // y >= height/2 (lower)
			pt.y = rec.y - 15;				// should be located on the top of the bouding box
			bg = cv::Rect(rec.x - 5, rec.y - 40, bgRecWidth, bgRecHeight);
		}
		cv::rectangle(img, bg, cv::Scalar(50, 50, 50), -1);
		cv::putText(img, strID, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255)/*color*/, thick);
	}
	cv::rectangle(img, rec, color, thick);
}
// Draw Frame Number on Image
void DrawFrameNumberAndFPS(int iFrameCnt, cv::Mat& img, double scale, int thick, int frameOffset, int frames_skip_interval, double sec) {
	// Draw Frame Number
	char frameCntBuf[8];
	sprintf_s(frameCntBuf, 8, "%d", (iFrameCnt + frameOffset) / frames_skip_interval);
	cv::putText(img, frameCntBuf, cv::Point(10, 65), CV_FONT_HERSHEY_SIMPLEX, scale, cvScalar(255, 255, 255), thick);

	// Draw Frames Per Second
	if (sec > 0.0) {
		string text = cv::format("%0.1f fps", 1.0 / sec * frames_skip_interval);
		cv::Scalar textColor(0, 0, 250);
		cv::putText(img, text, cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 2, textColor, 2);
	}
}
