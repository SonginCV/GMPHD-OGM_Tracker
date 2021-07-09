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
// GMPHD-OGM.h
#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <map>
#include <ppl.h>
#include <numeric>
#include <functional>

#include <boost\format.hpp>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\opencv.hpp>

#include "HungarianAlgorithm.h"

#define MAX_OBJECTS				27	
#define FRAME_OFFSET			1	

#define SIZE_CONSTRAINT_RATIO	2

#define PREDICTION_LEVEL_LOW	1
#define PREDICTION_LEVEL_MID	2

#define LOW_ASSOCIATION_DIMS	2

// Parameters for the GMPHD filter
#define PI						3.14159265
#define e						2.71828182
#define DIMS_STATE				6
#define DIMS_STATE_MID			4
#define DIMS_OBSERVATION		4
#define DIMS_OBSERVATION_MID	2
#define T_th					(0.0)
#define W_th					(0.0)
#define Q_TH_LOW_20				0.00000000000000000001f		// 0.1^20
#define Q_TH_LOW_20_INVERSE		100000000000000000000.0f	// 10^20
#define Q_TH_LOW_15				0.00000000000001f			// 0.1^15
#define Q_TH_LOW_15_INVERSE		1000000000000000.0f			// 10^15
#define Q_TH_LOW_10				0.0000000001f				// 0.1^10
#define Q_TH_LOW_10_INVERSE		10000000000.0f				// 10^10
#define Q_TH_LOW_8				0.00000001f					// 0.1^8
#define Q_TH_LOW_8_INVERSE		100000000.0f				// 10^8
#define P_SURVIVE_LOW			0.99						// object number >=2 : 0.99, else 0.95
#define P_SURVIVE_MID			0.95
#define VAR_X					25
#define VAR_Y					100
#define VAR_X_VEL				25
#define VAR_Y_VEL				100
#define VAR_WIDTH				100
#define VAR_HEIGHT				400
#define VAR_X_MID				100
#define VAR_Y_MID				400
#define VAR_X_VEL_MID			100
#define VAR_Y_VEL_MID			400

#define VELOCITY_UPDATE_ALPHA	0.95f

// Parameters for Data Association
#define TRACK_ASSOCIATION_FRAME_DIFFERENCE	0	// 0: equal or later, 1:later
#define ASSOCIATION_STAGE_1_GATING_ON		0
#define ASSOCIATION_STAGE_2_GATING_ON		0

// Parameters for Occlusion Group Management (Merge and Occlusion Group Energy Minimization)
#define MERGE_ON				1
#define MERGE_METRIC_OPT		0		// 0: SIOA, 1:IOU
#define MERGE_THRESHOLD_RATIO	0.5f
#define MERGE_METRIC_SIOA		0
#define MERGE_METRIC_IOU		1

#define GROUP_MANAGEMENT_FRAME_WISE_ON	1
		
#define USE_GMPHD_PLAIN			0
#define USE_GMPHD_HDA			1
#define USE_GMPHD_OGM			2
#define GMPHD_TRACKER_MODE		USE_GMPHD_OGM

#define DETECTION_FILTERING_ON	1

// Visualization Option
#define VISUALIZATION_MAIN_ON				0
#define SKIP_FRAME_BY_FRAME					0
#define BOUNDING_BOX_THICK					5
#define TRAJECTORY_THICK					5
#define	ID_CONFIDENCE_FONT_SIZE				2
#define	ID_CONFIDENCE_FONT_THICK			2
#define FRAME_COUNT_FONT_SIZE				3
#define FRAME_COUNT_FONT_THICK				3

typedef struct MOTparams {
	double DET_MIN_CONF = 0.0;
	int TRACK_MIN_SIZE = 1;
	int FRAMES_DELAY_SIZE = TRACK_MIN_SIZE-1;
	int T2TA_MAX_INTERVAL = 10;
	int MERGE_METRIC = MERGE_METRIC_OPT;
	float MERGE_RATIO_THRESHOLD = MERGE_THRESHOLD_RATIO;
	int GROUP_QUEUE_SIZE = TRACK_MIN_SIZE * 10;
	MOTparams() {
	}
	MOTparams(double dConf_th, int trk_min, int t2ta_max, int mg_metric, float mg_th, int group_q_size) {
		this->DET_MIN_CONF = dConf_th;
		this->TRACK_MIN_SIZE = trk_min;
		this->FRAMES_DELAY_SIZE = trk_min - 1;
		this->T2TA_MAX_INTERVAL = t2ta_max;
		this->MERGE_METRIC = mg_metric;
		this->MERGE_RATIO_THRESHOLD = mg_th;
		this->GROUP_QUEUE_SIZE = group_q_size;
	}
	MOTparams& operator=(const MOTparams& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->DET_MIN_CONF = copy.DET_MIN_CONF;
		this->TRACK_MIN_SIZE = copy.TRACK_MIN_SIZE;
		this->FRAMES_DELAY_SIZE = copy.FRAMES_DELAY_SIZE;
		this->T2TA_MAX_INTERVAL = copy.T2TA_MAX_INTERVAL;
		this->MERGE_METRIC = copy.MERGE_METRIC;
		this->MERGE_RATIO_THRESHOLD = copy.MERGE_RATIO_THRESHOLD;
		this->GROUP_QUEUE_SIZE = copy.GROUP_QUEUE_SIZE;

		return *this;
	}
} GMPHDOGMparams;

typedef struct boundingbox_id {
	int id;
	int idx;
	int min_id;		// minimum value ID in a group (is considered as group ID)
	double weight = 0.0;
	cv::Rect rec;	// t, t-1, t-2
	boundingbox_id() {
	}
	boundingbox_id(int id, cv::Rect occRect = cv::Rect()) :id(id) {
		// Deep Copy
		id = id;
		rec = occRect;
	}
	boundingbox_id& operator=(const boundingbox_id& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->idx = copy.idx;
		this->id = copy.id;
		this->min_id = copy.min_id;
		this->rec = copy.rec;
		this->weight = copy.weight;
	}
	bool operator<(const boundingbox_id& rect) const {
		return (id < rect.id);
	}
}RectID;

typedef struct bbTrack {
	int fn;
	int id;
	int id_associated = -1; // it is able to be used in Tracklet-wise association
	int fn_latest_T2TA = 0;
	cv::Rect rec;
	cv::Rect rec_t_1, rec_t_2;
	cv::Rect rec_corr;
	float vx, vx_prev;
	float vy, vy_prev;
	float weight;
	cv::Mat cov;
	cv::Mat tmpl;
	cv::Mat hist;
	float density;
	bool isAlive;
	bool isMerged = false;
	bool isInterpolated = false;
	int iGroupID = -1;
	bool isOcc = false;
	int size = 0;
	std::vector<RectID> occTargets;
	bbTrack() {}
	bbTrack(int fn, int id, int isOcc, cv::Rect rec, cv::Mat obj = cv::Mat(), cv::Mat hist = cv::Mat()) :
		fn(fn), id(id), isOcc(isOcc), rec(rec) {
		if (!hist.empty()) {
			this->hist.release();
			this->hist = hist.clone(); // deep copy
		}
		else {
			//this->hist.release();
			//printf("[ERROR]target_bb's parameter \"hist\" is empty!\n");
			this->hist = hist;
		}
		if (!obj.empty()) {
			this->tmpl.release();
			this->tmpl = obj.clone(); // deep copy
		}
		else {
			//this->obj_tmpl.release();
			this->tmpl = obj;
		}
		//isOccCorrNeeded = false; // default
	}
	bool operator<(const bbTrack& trk) const {
		return (id < trk.id);
	}
	bbTrack& operator=(const bbTrack& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;


		this->fn = copy.fn;
		this->id = copy.id;
		this->id_associated = copy.id_associated;
		this->size = copy.size;
		this->fn_latest_T2TA = copy.fn_latest_T2TA;
		this->rec = copy.rec;
		this->vx = copy.vx;
		this->vy = copy.vy;
		this->rec_t_1 = copy.rec_t_1;
		this->rec_t_2 = copy.rec_t_2;
		this->rec_corr = copy.rec_corr;
		this->vx_prev = copy.vx_prev;
		this->vy_prev = copy.vy_prev;
		this->density = copy.density;
		this->isAlive = copy.isAlive;
		this->isMerged = copy.isMerged;
		this->iGroupID = copy.iGroupID;
		this->isOcc = copy.isOcc;
		this->weight = copy.weight;

		if (!cov.empty()) this->cov = copy.cov.clone();
		if (!tmpl.empty()) this->tmpl = copy.tmpl.clone();
		if (!hist.empty()) this->hist = copy.hist.clone();

		return *this;
	}
	void CopyTo(bbTrack& dst) {
		dst.fn = this->fn;
		dst.id = this->id;
		dst.id_associated = this->id_associated;
		dst.rec = this->rec;
		dst.vx = this->vx;
		dst.vy = this->vy;
		dst.rec_t_1 = this->rec_t_1;
		dst.rec_t_2 = this->rec_t_2;
		dst.rec_corr = this->rec_corr;
		dst.vx_prev = this->vx_prev;
		dst.vy_prev = this->vy_prev;
		dst.density = this->density;
		dst.isAlive = this->isAlive;
		dst.isMerged = this->isMerged;
		dst.iGroupID = this->iGroupID;
		dst.isOcc = this->isOcc;
		dst.weight = this->weight;

		if (!this->cov.empty()) dst.cov = this->cov.clone();
		if (!this->tmpl.empty()) dst.tmpl = this->tmpl.clone();
		if (!this->hist.empty()) dst.hist = this->hist.clone();
	}
	void Destroy() {
		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	}

}BBTrk;
typedef struct bbDet {
	int fn;
	cv::Rect rec;
	float confidence;
	float weight; // normalization value of confidence at time t
	int id;// Used in Looking Back Association
}BBDet;

/**
* @brief	A Class for the GMPHD-OGM tracker
* @details	Input: images and object detection results / Output: tracking results of the GMPHD-PHD tracker
* @author	Young-min Song
* @date		2019-10-11
* @version	0.0.1
*/
class GMPHD_OGM
{
public:
	GMPHD_OGM();
	~GMPHD_OGM();
	void SetParams(GMPHDOGMparams params);
	void SetTotalFrames(int nFrames);
	GMPHDOGMparams GetParams();

	vector<vector<float>> DoMOT(int iFrmCnt, const cv::Mat& img, const vector<vector<float>> dets);

public:
	vector<vector<BBTrk>> allLiveReliables;

	cv::Mat *imgBatch;
	std::vector<BBDet> *detsBatch;

	cv::Scalar color_tab[MAX_OBJECTS];
private:
	GMPHDOGMparams params;
	int frmWidth, frmHeight;
	int iTotalFrames;
	int sysFrmCnt;
	int usedIDcnt;

	bool isInitialization;
	
	std::vector<BBTrk> liveTrkVec;		// live tracks at time t (now)
	std::vector<BBTrk> *liveTracksBatch;
	std::vector<BBTrk> lostTrkVec;		// lost tracks at time t (was live tracks before time t)
	std::vector<BBTrk> *lostTracksBatch;

	// THe container for Group management, index (0:t-d-2, 1:t-d-1, 2:t-d), d: delayed time
	std::map<int, std::vector<RectID>> *groupsBatch;

	std::map<int, std::vector<BBTrk>> tracksbyID;
	std::map<int, std::vector<BBTrk>> tracks_reliable;
	std::map<int, std::vector<BBTrk>> tracks_unreliable;

	HungarianAlgorithm HungAlgo;
private:
	// Init Containters, Matrices, and Utils
	void InitializeImagesQueue(int width, int height);
	void InitializeTrackletsContainters();
	void InitializeColorTab();
	void InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, int dims_state, int dims_obs);

	// Detection Filtering
	vector<BBDet> DetectionFilteringUsingConfidenceArea(vector<BBDet>& obs, const double T_merge = MERGE_THRESHOLD_RATIO, const double T_occ = 0.0);

	// Initialize states (tracks)
	void InitTracks(int iFrmCnt, const vector<BBDet> dets);

	// Prediction of state_k|k-1 from state_k-1 (x,y,vel_x,vel_y,width, height) using Kalman filter
	void PredictFrmWise(int iFrmCnt, vector<BBTrk>& stats, const cv::Mat F, const cv::Mat Q, cv::Mat &Ps, int iPredictionLevel);

	// Methods for D2T (FrmWise) and T2T (TrkWise) Association (Hierarchical Data Association)
	void DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, vector<BBDet>& obss, cv::Mat &Ps, const cv::Mat& H, double P_survive = 0.99, int offset = FRAME_OFFSET, int dims_low = LOW_ASSOCIATION_DIMS);
	void DataAssocTrkWise(int iFrmCnt, cv::Mat& img, vector<BBTrk>& stats_lost, vector<BBTrk>& obss_live);

	// Affinity (Cost) Calculation
	float FrameWiseAffinity(BBDet ob, BBTrk& stat_temp, const int dims = 2);
	float TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int& dims = 2);
	double GaussianFunc(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat);

	// Motion Estimation
	cv::Point2f LinearMotionEstimation(const vector<BBTrk>& tracklet, int& idx_first_last_fd, int& idx_first, int& idx_last, int reverse_offset = 0, int required_Q_size = 0);

	// Calculate the minimum cost pairs by Hungarian method (locally optimized)
	std::vector<vector<int>> HungarianMethod(int* r, int nObs, int nStats, int max_cost = 0);
	std::vector<std::vector<double> > array_to_matrix_dbl(int* m, int rows, int cols, int max_cost = 0);

	// Functions for Occlusion Group Management
	/// Merge
	void CheckOcclusionsMergeStates(vector<BBTrk>& stats, const double T_merge = MERGE_THRESHOLD_RATIO, const double T_occ = 0.0);
	int  FindMinIDofNeigbors2Depth(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id);		// Only 2 depth search
	/// OGEM
	void CheckOcclusionsGroups(vector<BBTrk>& stats, const double T_merge = 1.0, const double T_occ = 0.0);
	void UnifyNeighborGroups(vector<BBTrk> input_targets);
	vector<double[2]> MinimizeGroupCost(int iFrmCnt, int group_min_id, cv::Rect group_rect, vector<RectID>* objs, vector<RectID> groupRects, vector<BBTrk>& liveReliables);
	cv::Rect CalGroupRect(int group_min_id, vector<RectID> groupRects);
	void CvtRec2Mat4TopologyModel(int dims, cv::Rect origin, cv::Rect target, cv::Mat& m);

	// Tracklets Managements (Categorization, state transition, memory deallocation..)
	void ArrangeTargetsVecsBatchesLiveLost();
	void PushTargetsVecs2BatchesLiveLost();
	void SortTrackletsbyID(map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& targets);
	void ClassifyTrackletReliability(int iFrmCnt, map<int, vector<BBTrk>>& tracksbyID, map<int, vector<BBTrk>>& reliables, map<int, std::vector<BBTrk>>& unreliables);
	void ClassifyReliableTracklets2LiveLost(int iFrmCnt, const map<int, vector<BBTrk>>& reliables, vector<BBTrk>& liveReliables, vector<BBTrk>& LostReliables, vector<BBTrk>& obss);
	void ArrangeRevivedTracklets(map<int, vector<BBTrk>>& tracks, vector<BBTrk>& lives);
	void ClearOldEmptyTracklet(int current_fn, map<int, vector<BBTrk>>& tracklets, int MAXIMUM_OLD);

	// Return the Tracking Results
	/// Return live and reliable tracks
	vector<vector<float>> ReturnTrackingResults(int iFrmCnt, vector<BBTrk>& liveReliables);

	// Utils
	bool IsOutOfFrame(cv::Rect rec, int fWidth, int fHeight);
	float CalcIOU(cv::Rect A, cv::Rect B);
	float CalcSIOA(cv::Rect A, cv::Rect B);
	inline cv::Rect cvMergeRects(const cv::Rect rect1, const cv::Rect rect2, double alpha);

private:
	cv::Mat F;			// transition matrix state_t-1 to state_t 	
	cv::Mat Q;			// process noise covariance
	cv::Mat Ps;			// covariance of states's Gaussian Mixtures for Survival
	cv::Mat R;			// the covariance matrix of measurement
	cv::Mat H;			// transition matrix state_t to observation_t

	cv::Mat F_mid;		// transition matrix state_t-1 to state_t 	
	cv::Mat Q_mid;		// process noise covariance
	cv::Mat Ps_mid;		// covariance of states's Gaussian Mixtures for Survival
	cv::Mat R_mid;		// the covariance matrix of measurement
	cv::Mat H_mid;		// transition matrix state_t to observation_t
	cv::Mat R_mid_v;	// the covariance matrix of measurement for T2TA_wt_velocity
	cv::Mat H_mid_v;	// transition matrix state_t to observation_t for T2TA_wt_velocity

	double P_survive = P_SURVIVE_LOW;		// Probability of Survival	(User Parameter)(Constant)
	double P_survive_mid = P_SURVIVE_MID;
};

