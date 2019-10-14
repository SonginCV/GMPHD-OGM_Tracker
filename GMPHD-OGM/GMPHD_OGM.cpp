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
// GMPHD-OGM.cpp

#include "stdafx.h"
#include "GMPHD_OGM.h"

GMPHD_OGM::GMPHD_OGM()
{
	this->iTotalFrames = 0;
	this->sysFrmCnt = 0;
	this->usedIDcnt = 0;

	this->isInitialization = false;

	this->InitializeColorTab();

	//this->InitializeTrackletsContainters(); // called in SetParams() again

	this->InitializeMatrices(F, Q, Ps, R, H, DIMS_STATE, DIMS_OBSERVATION);
	this->InitializeMatrices(F_mid, Q_mid, Ps_mid, R_mid, H_mid, DIMS_STATE_MID, DIMS_OBSERVATION_MID);
}


GMPHD_OGM::~GMPHD_OGM()
{
}
vector<vector<float>> GMPHD_OGM::DoMOT(int iFrmCnt, const cv::Mat& img, const vector<vector<float>> dets) {
	if (this->sysFrmCnt == 0)
		InitializeImagesQueue(img.cols, img.rows);

	// Load Detection (truncating and filtering)
	std::vector<BBDet> detVec;
	int nDets = dets.size();
	for (int d = 0; d < nDets; ++d) {
		BBDet bbd;
		bbd.fn = iFrmCnt; // (int)dets[d][0];
		bbd.rec.x = (int)dets[d][1];
		bbd.rec.y = (int)dets[d][2];
		bbd.rec.width = (int)dets[d][3];
		bbd.rec.height = (int)dets[d][4];
		bbd.confidence = dets[d][5];

		//printf("[%d](%d,%d,%d,%d,%lf)\n",iFrmCnt,bbd.rec.x,bbd.rec.y,bbd.rec.width,bbd.rec.height,bbd.confidence);
		if (bbd.confidence >= this->params.DET_MIN_CONF) {

			if(bbd.confidence < 0.0) 
				bbd.confidence = 0.001; // 1.0 / nDets;
			detVec.push_back(bbd);
		}
	}
	if (DETECTION_FILTERING_ON) {
		std::vector<BBDet> filteredDetVec = this->DetectionFilteringUsingConfidenceArea(detVec);
		detVec.clear();
		detVec = filteredDetVec;
	}

	// Normalize the Weights (Detection Scores)
	vector<BBDet>::iterator iterDet;
	double sumConf = 0.0;
	for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
		sumConf += iterDet->confidence;
	}
	if (sumConf > 0.0) {
		for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
			iterDet->weight = iterDet->confidence / sumConf;
		}
	}
	else if (sumConf <= 0.0) {
		for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
			iterDet->weight = 0.001;
		}
	}

	// Keep the images and observations into vector array within the recent this->params.TRACK_MIN_SIZE frames 
	if (this->sysFrmCnt >= this->params.TRACK_MIN_SIZE) {

		for (int q = 0; q < this->params.FRAMES_DELAY_SIZE; q++) {
			imgBatch[q + 1].copyTo(imgBatch[q]);
		}
		img.copyTo(imgBatch[this->params.FRAMES_DELAY_SIZE]);
		detsBatch[this->params.FRAMES_DELAY_SIZE] = detVec;
	}
	else if (this->sysFrmCnt < this->params.TRACK_MIN_SIZE) {
		img.copyTo(imgBatch[this->sysFrmCnt]);
		detsBatch[this->sysFrmCnt] = detVec;
	}

	/*-------------------- Stage 1. Detection-to-Track Association (D2TA) --------------------*/
	/// Init, Predict -> Data Association -> Update -> Pruning -> Merge
	if (!this->liveTrkVec.empty()) {
		// Predict
		this->PredictFrmWise(iFrmCnt, liveTrkVec, F, Q, Ps, PREDICTION_LEVEL_LOW);
		// Data Association -> Update -> Pruning -> Merge
		this->DataAssocFrmWise(iFrmCnt, img, this->liveTrkVec, detVec, this->Ps, this->H, this->P_survive, FRAME_OFFSET);
	}
	else if (this->sysFrmCnt == 0 || this->liveTrkVec.size() == 0) {
		// Init
		this->InitTracks(iFrmCnt, detVec);
	}
	// Arrange the targets which have been alive or not (live or lost)
	this->ArrangeTargetsVecsBatchesLiveLost();

	// Push the Tracking Results (live, lost) into the each Tracks Queue (low level)
	/// Keep only the tracking targets at now (except the loss targets)
	this->PushTargetsVecs2BatchesLiveLost();

	if (GMPHD_TRACKER_MODE) {

		// Tracklet Categorization (live or lost) for Track-to-Track Association (T2TA)
		// A Reliable Tracklet has the size (length) >= params.TRACK_MIN_SIZE.
		// A Unreliable Tracklet has the size < params.TRAK_MIN_SIZE.
		// Only the Reliable Tracklets are used for T2TA
		/// Put the re-arranged targets into tracklets according to ID, frame by frame
		/// Insert the re-arranged tracklets to tracklets map according to ID as a key
		this->SortTrackletsbyID(this->tracksbyID, this->liveTrkVec);
		this->ClassifyTrackletReliability(iFrmCnt, this->tracksbyID, this->tracks_reliable, this->tracks_unreliable);

		vector<BBTrk> liveReliables, lostReliables, obss;
		this->ClassifyReliableTracklets2LiveLost(iFrmCnt, this->tracks_reliable, liveReliables, lostReliables, obss);

		/*-------------------- Stage 2. Track-to-Track Association (T2TA) --------------------*/
		/// Data Association -> Update Tracklets -> Occlusion Group Enerege Minimization (OGEM)
		if (!lostReliables.empty() && /*!obss.empty()*/ !liveReliables.empty()) {

			cv::Mat img_latency = imgBatch[0].clone();	// image for no latent association tracking
			this->DataAssocTrkWise(iFrmCnt - this->params.FRAMES_DELAY_SIZE, img_latency, lostReliables, liveReliables);
			//this->DataAssocTrkWise(iFrmCnt - this->params.FRAMES_DELAY_SIZE, img_latency, lostReliables, obss);

			this->ArrangeRevivedTracklets(this->tracks_reliable, liveReliables); // it can be tracks_unreliabe, liveUnreliables
			//this->ArrangeRevivedTracklets(this->tracks_reliable, obss);
			img_latency.release();
		}
		if (!obss.empty())
			liveReliables.insert(std::end(liveReliables), std::begin(obss), std::end(obss));

		/*-------------------- Occlusion Group Enerege Minimization (OGEM) --------------------*/
		this->CheckOcclusionsGroups(liveReliables, 1.0, 0.0); // do only occlusion check
		this->UnifyNeighborGroups(liveReliables);

		vector<BBTrk>::iterator iterR;
		for (iterR = liveReliables.begin(); iterR != liveReliables.end(); ++iterR) {
			if (!iterR->occTargets.empty()) // linear motion 으로 해볼까
			{
				vector<BBTrk> liveTrk = this->tracks_reliable[iterR->id];
				int a_trk_fd = 0;
				int idx1 = 0, idx2 = 0; // actural index of first and last in object bbox queue
				int rOffset = 2;
				int qSize = 0;
				cv::Point2f v = this->LinearMotionEstimation(liveTrk, a_trk_fd, idx1, idx2, rOffset, qSize);
				cv::Rect rec_corrected = liveTrk[idx2].rec;
				if (v.x == 0.0 && v.y == 0.0 && a_trk_fd == 0) {
					// Do not correct
					rec_corrected = iterR->rec;
				}
				else {
					// correct bbox by using the motion between two bboxes in the previous frame interval
					int passed_frames = liveTrk.back().fn - liveTrk[idx2].fn;
					rec_corrected.x += ((float)passed_frames * v.x);
					rec_corrected.y += ((float)passed_frames * v.y);
				}
				iterR->rec_corr = rec_corrected;
				this->tracks_reliable[iterR->id].back().rec_corr = iterR->rec_corr;
			}
		}
		map<int, vector<RectID>>::iterator iterG;
		for (iterG = this->groupsBatch[this->params.GROUP_QUEUE_SIZE - 1].begin(); iterG != this->groupsBatch[this->params.GROUP_QUEUE_SIZE - 1].end(); ++iterG) {
			if (!iterG->second.empty()) {

				// Union All Objects' Rects in a Group
				int group_min_id = iterG->second.front().min_id; // index of group map
				cv::Rect group_rect = this->CalGroupRect(group_min_id, iterG->second);
				int GROUP_OBJECTS = iterG->second.size();
				int OBJECTS_COMB = GROUP_OBJECTS*(GROUP_OBJECTS - 1) / 2;
				vector<double[2]> gCosts;
				vector<RectID> objs[2];
				double min_cost = DBL_MAX;
				int min_hIdx = 0;

				if (GROUP_OBJECTS == 2 || GROUP_OBJECTS == 3) {

					// Cacluate Cosine Similarity of the Two Objects' Relative Motions in a Group
					// & Correct ID-swicth by using Cosine Simliarity [-1.0,1.0] = +1.0, /2.0 => Group Cost [0.0,1.0]
					gCosts = this->MinimizeGroupCost(iFrmCnt, group_min_id, group_rect, objs, iterG->second, liveReliables);
					for (int h = 0; h < gCosts.size(); ++h) {
						if (min_cost > gCosts[h][0]) {
							min_cost = gCosts[h][0];
							min_hIdx = h;
						}
					}

					// Scaling the group cost [0.0,1.0]
					for (int h = 0; h < gCosts.size(); ++h) {
						double tCost = gCosts[h][0];
						if (tCost > 1000.0) tCost = 0.0;
						else {
							tCost = 1.0 - tCost / 200.0;   // alpha
							if (tCost < 0.0) tCost = 0.0;
						}
						gCosts[h][0] = tCost;

						double tCost2 = gCosts[h][1];
						if (tCost2 > 1000.0) tCost2 = 0.0;
						else {
							tCost2 = 1.0 - tCost2 / 200.0;   // alpha
							if (tCost2 < 0.0) tCost2 = 0.0;
						}
						gCosts[h][1] = tCost2;
					}
				}
			}
		}

		// Tracklets' Containters Management (for practical implementation)
		if ((this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE > this->params.T2TA_MAX_INTERVAL) && (((this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE) % this->params.T2TA_MAX_INTERVAL) == 0)) {

			//cout << "this->tracksbyID: ";
			ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracksbyID, this->params.T2TA_MAX_INTERVAL);
			//cout << "this->tracks_reliable: ";
			ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracks_reliable, this->params.T2TA_MAX_INTERVAL);
			//cout << "this->tracks_unreliable: ";
			if (this->params.TRACK_MIN_SIZE > 1)
				ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracks_unreliable, this->params.T2TA_MAX_INTERVAL);

			/*vector<BBTrk> tLiveReliables, tLostReliables;
			this->ClassifyReliableTracklets2LiveLost(iFrmCnt, this->tracks_reliable, tLiveReliables, tLostReliables);

			printf("lost (%d->%d) and live (%d->%d) tracklets.\n", lostReliables.size(), tLostReliables.size(), liveReliables.size(), tLiveReliables.size());

			tLiveReliables.clear();
			tLostReliables.clear();*/
		}


		// Return the Tracking Results in frame iFrmCnt
		vector<vector<float>> tracks = this->ReturnTrackingResults(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, liveReliables/*this->liveTrkVec*/);
		liveReliables.clear();
		lostReliables.clear();
		obss.clear();
		this->sysFrmCnt++;
		return tracks;
	}
	else {
		// Return the Tracking Results in frame iFrmCnt
		vector<vector<float>> tracks = this->ReturnTrackingResults(this->sysFrmCnt, this->liveTrkVec);
		this->sysFrmCnt++;
		return tracks;
	}
}
void GMPHD_OGM::InitTracks(int iFrmCnt, const vector<BBDet> dets) {
	std::vector<BBDet>::const_iterator iterD;
	for (iterD = dets.begin(); iterD != dets.end(); ++iterD)
	{
		int id = this->usedIDcnt++;

		BBTrk bbt;
		bbt.isAlive = true;
		bbt.id = id;
		bbt.fn = iFrmCnt;
		bbt.rec = iterD->rec;
		bbt.vx = 0.0;
		bbt.vy = 0.0;
		bbt.weight = iterD->weight;

		bbt.cov = (cv::Mat_<double>(4, 4) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0,
			0, 0, VAR_X_VEL, 0,
			0, 0, 0, VAR_Y_VEL);

		this->liveTrkVec.push_back(bbt);
	}
}
void GMPHD_OGM::PredictFrmWise(int iFrmCnt, vector<BBTrk>& stats, const cv::Mat F, const cv::Mat Q, cv::Mat &Ps, int iPredictionLevel)
{
	int dims_state = stats.at(0).cov.cols;
	int dims_obs = stats.at(0).cov.cols - 2;

	if (iPredictionLevel == PREDICTION_LEVEL_LOW) {			// low level prediction

		vector<BBTrk>::iterator iter;

		for (iter = stats.begin(); iter < stats.end(); ++iter) {

			iter->fn = iFrmCnt;

			if (iPredictionLevel == PREDICTION_LEVEL_LOW) {
				if (dims_state == 4)
				{
					cv::Mat Ps_temp = Q_mid + F_mid*iter->cov*F_mid.t();

					// make covariance matrix diagonal
					//Ps_temp.copyTo(iter->cov); 
					iter->cov.at<double>(0, 0) = Ps_temp.at<double>(0, 0);
					iter->cov.at<double>(1, 1) = Ps_temp.at<double>(1, 1);
					iter->cov.at<double>(2, 2) = Ps_temp.at<double>(2, 2);
					iter->cov.at<double>(3, 3) = Ps_temp.at<double>(3, 3);
				}
				if (dims_state == 6)
				{
					cv::Mat Ps_temp = Q + F*iter->cov*F.t();
					Ps_temp.copyTo(iter->cov);
				}
			}

			// copy current stat bounding box and velocity info to previous stat
			iter->vx_prev = iter->vx;
			iter->vy_prev = iter->vy;

			iter->rec_t_2 = iter->rec_t_1;
			iter->rec_t_1 = iter->rec;

			iter->rec.x += iter->vx; // Same as xk|k-1=F*xk-1
			iter->rec.y += iter->vy;
		}
	}
}
void GMPHD_OGM::DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, vector<BBDet>& obss, cv::Mat &Ps, const cv::Mat& H, double P_survive, int offset, int dims_low) {

	int nObs = obss.size();
	int mStats = stats.size();
	int* m_cost = new int[nObs*mStats];
	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(mStats, 0.0));
	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(nObs, std::vector<BBTrk>(mStats, BBTrk()));
	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c)
		{
			stats.at(c).CopyTo(stats_matrix[r][c]);
		}
	}

	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < nObs; ++r){
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			// Calculate the Affinity between detection (observations) and tracking (states)
			q_values[r][c] = FrameWiseAffinity(obss[r], stats_matrix[r][c], 2);

			if (ASSOCIATION_STAGE_1_GATING_ON) {
				if (q_values[r][c] < Q_TH_LOW_20) {
					q_values[r][c] = 0.0;
					float overlapping_ratio = 0.0;
					if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)
						overlapping_ratio = this->CalcIOU(obss[r].rec, stats_matrix[r][c].rec);
					else
						overlapping_ratio = this->CalcSIOA(obss[r].rec, stats_matrix[r][c].rec);

					if (overlapping_ratio >= this->params.MERGE_RATIO_THRESHOLD) // threshold
						q_values[r][c] = obss[r].weight*overlapping_ratio;
				}
			}
		}
	}
	);

	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	//Concurrency::parallel_for(0, nObs, [&](int r) {
		for (int r = 0; r < nObs; ++r){
		int nStats = stats_matrix[r].size();
		double denominator = 0.0;
		for (int l = 0; l < stats_matrix[r].size(); ++l)
			denominator += (stats_matrix[r][l].weight * q_values[r][l]);

		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			double numerator =  /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];
			stats_matrix[r][c].weight = numerator / denominator;							// (19), numerator(분자), denominator(분모)

			// Scaling the affinity value to Integer
			if (stats_matrix[r][c].weight > 0.0) {
					if ((double)(stats_matrix[r][c].weight) < (double)(FLT_MIN) / (double)10.0) {
						std::cerr << "[" << iFrmCnt << "] weight(FW) < 0.1*FLT_MIN" << std::endl;
						m_cost[r*nStats + c] = 9999; // log(1.0)
					}
					else m_cost[r*nStats + c] = -100.0*log2l((double)numerator);
			}
			else {
				m_cost[r*nStats + c] = 10000;
			}
		}
	}
	//);
	int min_cost = INT_MAX, max_cost = 0;
	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c) {
			if (min_cost > m_cost[r*mStats + c]) min_cost = m_cost[r*mStats + c];
			if (max_cost < m_cost[r*mStats + c]) max_cost = m_cost[r*mStats + c];
		}
	}

	// Solving data association problem (=linear assignment problem) (find the min cost assignment pairs)
	std::vector<vector<int>> assigns;
	assigns = this->HungarianMethod(m_cost, obss.size(), stats.size(), max_cost);

	bool *isAssignedStats = new bool[stats.size()];	memset(isAssignedStats, 0, stats.size());
	bool *isAssignedObs = new bool[obss.size()];	memset(isAssignedObs, 0, obss.size());
	int *isAssignedObsIDs = new int[stats.size()];	memset(isAssignedObsIDs, 0, stats.size()); // only used in LB_ASSOCIATION

	for (int c = 0; c < stats.size(); ++c) {
		for (int r = 0; r < obss.size(); ++r) {
			if (assigns[r][c] == 1 && m_cost[r*stats.size() + c] < max_cost) {

				// Velocity Update
				float vx_t_1 = stats[c].vx;
				float vy_t_1 = stats[c].vy;

				float vx_t = (obss[r].rec.x + obss[r].rec.width / 2.0) - (stats[c].rec.x + stats[c].rec.width / 2.0);
				float vy_t = (obss[r].rec.y + obss[r].rec.height / 2.0) - (stats[c].rec.y + stats[c].rec.height / 2.0);

				stats[c].vx = vx_t_1*VELOCITY_UPDATE_ALPHA + vx_t*(1.0 - VELOCITY_UPDATE_ALPHA);
				stats[c].vy = vy_t_1*VELOCITY_UPDATE_ALPHA + vy_t*(1.0 - VELOCITY_UPDATE_ALPHA);

				stats[c].weight = stats_matrix[r][c].weight;

				stats[c].rec = obss[r].rec;

				// Covariance Matrix Update
				stats_matrix[r][c].cov.copyTo(stats[c].cov);

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;
				isAssignedObsIDs[c] = obss[r].id; // only used in LB_ASSOCIATION
				break;
			}
			isAssignedStats[c] = false;
		}
		stats[c].isAlive = isAssignedStats[c];

	}

	// Weight Normalization after GMPHD association process
	double sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		sumWeight += stats[c].weight;
	}
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;

		}
	}

	vector<int> newTracks;
	for (int r = 0; r < obss.size(); ++r) {
		if (!isAssignedObs[r]) {
			newTracks.push_back(r);
			BBTrk newTrk;
			newTrk.fn = iFrmCnt;
			newTrk.id = this->usedIDcnt++;
			newTrk.cov = (cv::Mat_<double>(4, 4) << \
				VAR_X, 0, 0, 0, \
				0, VAR_Y, 0, 0,
				0, 0, VAR_X_VEL, 0,
				0, 0, 0, VAR_Y_VEL);
			newTrk.rec = obss[r].rec;
			newTrk.isAlive = true;
			newTrk.vx = 0.0;
			newTrk.vy = 0.0;
			newTrk.weight = obss[r].weight;

			stats.push_back(newTrk);
		}
	}
	if (MERGE_ON) {
		this->CheckOcclusionsMergeStates(stats, this->params.MERGE_RATIO_THRESHOLD, 0.0);
	}

	// Weight Normalization After Birth Processing
	sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			sumWeight += stats[c].weight;
		}
	}
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;
		}
	}

	/// Memory Deallocation
	delete[]isAssignedStats;
	delete[]isAssignedObs;
	delete[]isAssignedObsIDs;
	delete[]m_cost;
}
vector<vector<int>> GMPHD_OGM::HungarianMethod(int* r, int nObs, int nStats, int max_cost) {

	std::vector< std::vector<double> > costMatrix = array_to_matrix_dbl(r, nObs, nStats, max_cost);

	vector<vector<int>> assigns;
	assigns.resize(nObs, std::vector<int>(nStats, 0));

	vector<int> assignment;

	double cost = this->HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++) {
		if (assignment[x] >= 0)
			assigns[assignment[x]][x] = 1;
	}
	return assigns;
}
vector< std::vector<double> >  GMPHD_OGM::array_to_matrix_dbl(int* m, int rows, int cols, int max_cost) {
	int i, j;
	int rows_hung = cols;
	int cols_hung = rows;
	std::vector< std::vector<double> > r;
	r.resize(rows_hung, std::vector<double>(cols_hung, max_cost));

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) {
			r[j][i] = m[i*cols + j];
		}
	}
	return r;
}
float GMPHD_OGM::FrameWiseAffinity(BBDet ob, BBTrk &stat_temp, const int dims_obs) {

	// Bounding box size contraint
	if ((stat_temp.rec.area() >= ob.rec.area() * SIZE_CONSTRAINT_RATIO) || (stat_temp.rec.area() * SIZE_CONSTRAINT_RATIO <= ob.rec.area())) return 0.0;

	// Bounding box location contraint(gating)
	if (stat_temp.rec.area() >= ob.rec.area()) {
		if ((stat_temp.rec & ob.rec).area() < ob.rec.area() / 2) return 0.0;
	}
	else {
		if ((stat_temp.rec & ob.rec).area() < stat_temp.rec.area() / 2) return 0.0;
	}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;
	int dims_stat = dims_obs + 2;
	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	// (23)
	cv::Mat z_cov_temp(dims_obs, dims_obs, CV_64FC1);
	//z_cov_temp = H*Ps*H.t() + R;

	z_cov_temp = H_mid*stat_temp.cov*H_mid.t() + R_mid;

	//K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	K = stat_temp.cov*H_mid.t()*z_cov_temp.inv(cv::DECOMP_SVD);
	// (22)
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);
	//Ps_temp = Ps - K*H*Ps;
	Ps_temp = stat_temp.cov - K*H_mid*stat_temp.cov;
	Ps_temp.copyTo(stat_temp.cov);

	cv::Mat z_temp(dims_obs, 1, CV_64FC1);
	z_temp = (cv::Mat_<double>(dims_obs, 1) << ob.rec.x + (double)ob.rec.width / 2.0, ob.rec.y + (double)ob.rec.height / 2.0/*, ob.rec.width, ob.rec.height*/);

	// (20)
	cv::Mat mean_obs(dims_obs, 1, CV_64FC1);
	mean_obs.at<double>(0, 0) = (double)stat_temp.rec.x + (double)stat_temp.rec.width / 2.0;
	mean_obs.at<double>(1, 0) = (double)stat_temp.rec.y + (double)stat_temp.rec.height / 2.0;

	q_value = this->GaussianFunc(dims_obs, z_temp, mean_obs, z_cov_temp);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
float GMPHD_OGM::TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int& dims_obs) {

	// Bounding box size contraint
	if ((stat_pred.rec.area() >= obs.rec.area() * SIZE_CONSTRAINT_RATIO) || (stat_pred.rec.area() * SIZE_CONSTRAINT_RATIO <= obs.rec.area())) return 0.0;

	// Bounding box location contraint(gating)
	if (stat_pred.rec.area() >= obs.rec.area()) {
		if ((stat_pred.rec & obs.rec).area() <= 0 /*obs.rec.area() / 4*/) return 0.0;
	}
	else {
		if ((stat_pred.rec & obs.rec).area() <= 0  /*stat_pred.rec.area() / 4*/) return 0.0;
	}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;
	int dims_stat = dims_obs + 2;
	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	// (23)
	cv::Mat z_cov_temp(dims_obs, dims_obs, CV_64FC1);
	//z_cov_temp = H*Ps*H.t() + R;

	z_cov_temp = H_mid*stat_pred.cov*H_mid.t() + R_mid;

	//K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	K = stat_pred.cov*H_mid.t()*z_cov_temp.inv(cv::DECOMP_SVD);
	// (22)
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);
	//Ps_temp = Ps - K*H*Ps;
	Ps_temp = stat_pred.cov - K*H_mid*stat_pred.cov;
	Ps_temp.copyTo(stat_pred.cov);

	cv::Mat z_temp(dims_obs, 1, CV_64FC1);
	z_temp = (cv::Mat_<double>(dims_obs, 1) << obs.rec.x + (double)obs.rec.width / 2.0, obs.rec.y + (double)obs.rec.height / 2.0/*, obs_live.vx, obs_live.vy*/);

	// (20)
	// H*GMM[i] : k-1 와 k-2 사이의 속도로 추론한 state를 Observation으로 transition
	// width 와 height에는 속도 미적용
	cv::Mat mean_obs(dims_obs, 1, CV_64FC1);
	mean_obs.at<double>(0, 0) = (double)stat_pred.rec.x + (double)stat_pred.rec.width / 2.0;
	mean_obs.at<double>(1, 0) = (double)stat_pred.rec.y + (double)stat_pred.rec.height / 2.0;
	//mean_obs.at<double>(2, 0) = (double)stat_temp.vx;
	//mean_obs.at<double>(3, 0) = (double)stat_temp.vy;

	q_value = this->GaussianFunc(dims_obs, z_temp, mean_obs, z_cov_temp);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
double GMPHD_OGM::GaussianFunc(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat) {
	double probability = -1.0;
	if ((x.rows != m.rows) || (cov_mat.rows != cov_mat.cols) || (x.rows != D)) {
		printf("[ERROR](x.rows!=m.rows) || (cov_mat.rows!=cov_mat.cols) || (x.rows!=D) (line:258)\n");
	}
	else {
		cv::Mat sub(D, 1, CV_64FC1);
		cv::Mat power(1, 1, CV_64FC1);
		double exponent = 0.0;
		double coefficient = 1.0;

		sub = x - m;
		power = sub.t() * cov_mat.inv(cv::DECOMP_SVD) * sub;

		coefficient = ((1.0) / (pow(2.0*PI, (double)D / 2.0)*pow(cv::determinant(cov_mat), 0.5)));
		exponent = (-0.5)*(power.at<double>(0, 0));
		probability = coefficient*pow(e, exponent);

		sub.release();
		power.release();
	}
	if (probability < Q_TH_LOW_15) probability = 0.0;

	//if (0) { // GpuMat
	//	cv::cuda::GpuMat subGpu;
	//	cv::cuda::GpuMat powerGpu;
	//}
	return probability;
}
cv::Point2f GMPHD_OGM::LinearMotionEstimation(vector<BBTrk> tracklet, int& idx1_2_fd, int& idx1, int& idx2, int reverse_offset, int required_Q_size) {
	int idx_last, idx_first;
	
	int T_SIZE = tracklet.size();
	if ((T_SIZE - 1 - reverse_offset) >= 0)	idx_last = (T_SIZE - 1 - reverse_offset);
	else									idx_last = (T_SIZE - 1);

	if (!required_Q_size) idx_first = 0;
	else {
		if (idx_last >= (required_Q_size - 1))	idx_first = (idx_last - required_Q_size + 1);
		else									idx_first = 0;
	}
	idx1 = idx_first;
	idx2 = idx_last;

	float fd = tracklet[idx_last].fn - tracklet[idx_first].fn;
	idx1_2_fd = (int)fd;

	cv::Rect r1, r2;
	cv::Point2f cp1, cp2;
	cv::Point2f v;

	r1 = tracklet[idx_first].rec;
	r2 = tracklet[idx_last].rec;

	cp1 = cv::Point2f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0);
	cp2 = cv::Point2f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0);

	if (fd > 0) { // idx_first < idx_last
		v.x = (cp2.x - cp1.x) / fd;
		v.y = (cp2.y - cp1.y) / fd;
	}
	else {
		v.x = 0.0;
		v.y = 0.0;
	}

	return v;
}
cv::Point2f GMPHD_OGM::LinearMotionEstimation(map<int, vector<BBTrk>> tracks, int id, int &idx1_2_fd, int& idx1, int& idx2, int reverse_offset, int required_Q_size) {

	int idx_last, idx_first;
	int T_SIZE = tracks[id].size();
	if ((T_SIZE - 1 - reverse_offset) >= 0)	idx_last = (T_SIZE - 1 - reverse_offset);
	else									idx_last = (T_SIZE - 1);

	if (!required_Q_size) idx_first = 0;
	else {
		if (idx_last >= (required_Q_size - 1))	idx_first = (idx_last - required_Q_size + 1);
		else									idx_first = 0;
	}
	idx1 = idx_first;
	idx2 = idx_last;

	float fd = tracks[id][idx_last].fn - tracks[id][idx_first].fn;
	idx1_2_fd = (int)fd;

	cv::Rect r1, r2;
	cv::Point2f cp1, cp2;
	cv::Point2f v;

	r1 = tracks[id][idx_first].rec;
	r2 = tracks[id][idx_last].rec;

	cp1 = cv::Point2f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0);
	cp2 = cv::Point2f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0);

	if (fd > 0) { // idx_first < idx_last
		v.x = (cp2.x - cp1.x) / fd;
		v.y = (cp2.y - cp1.y) / fd;
	}
	else {
		v.x = 0.0;
		v.y = 0.0;
	}

	return v;
}
void GMPHD_OGM::DataAssocTrkWise(int iFrmCnt, cv::Mat& img, vector<BBTrk>& stats_lost, vector<BBTrk>& obss_live) {
	double min_cost_dbl = DBL_MAX;
	int nObs = obss_live.size();
	int mStats = stats_lost.size();
	int* m_cost = new int[nObs*mStats];

	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(mStats, 0.0));
	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(nObs, std::vector<BBTrk>(mStats, BBTrk()));

	for (int r = 0; r < obss_live.size(); ++r) {
		//stats.assign(stats_matrix[r].begin(), stats_matrix[r].end());
		for (int c = 0; c < stats_lost.size(); ++c)
		{
			stats_lost.at(c).CopyTo(stats_matrix[r][c]);
		}
	}

	Concurrency::parallel_for(0, nObs, [&](int r) {

		for (int c = 0; c < stats_matrix[r].size(); ++c) {

			int lostID = stats_matrix[r][c].id;
			int liveID = obss_live.at(r).id;

			float fd = this->tracks_reliable[liveID].front().fn - this->tracks_reliable[lostID].back().fn;

			double Taff = 1.0;
			int trk_fd = 0; // frame_difference
			if (fd >= TRACK_ASSOCIATION_FRAME_DIFFERENCE && fd < this->params.T2TA_MAX_INTERVAL) { // ==0 일때는 occlusion 을 감안해야 할듯 일단 >0 으로 해보자

																								   // Linear Motion Estimation
				int idx_first = 0;
				int idx_last = this->tracks_reliable[lostID].size() - 1;
				cv::Point2f v = this->LinearMotionEstimation(this->tracks_reliable, lostID, trk_fd, idx_first, idx_last);

				BBTrk stat_pred;
				this->tracks_reliable[lostID].back().CopyTo(stat_pred);

				stat_pred.vx = v.x;
				stat_pred.vy = v.y;

				//printf("(%lf,%lf)\n",v.x,v.y);

				stat_pred.rec.x = stat_pred.rec.x + stat_pred.vx*fd;
				stat_pred.rec.y = stat_pred.rec.y + stat_pred.vy*fd;

				//this->cvBoundingBox(img, obs_pred.rec, this->color_tab[obs_pred.id % 26], 3);

				if (this->IsOutOfFrame(stat_pred.rec, this->frmWidth, this->frmHeight))
					q_values[r][c] = 0;
				else
					q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinity(stat_pred, this->tracks_reliable[liveID].front(), 2);
				//q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinityVelocity(stat_pred, this->tracks_reliable[liveID].front(), 4);
				//-> this is too sensitive to deal with unstable detection bounding boxes

			}
			else
				q_values[r][c] = 0;
			// Calculate the Affinity between detection (observations) and tracking (states)
		}
	}
	);
	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < nObs; ++r){
		int nStats = stats_matrix[r].size();
		// (19)
		double denominator = 0.0;													// (19)'s denominator(분모)
		for (int l = 0; l < stats_matrix[r].size(); ++l) {

			denominator += (stats_matrix[r][l].weight * q_values[r][l]);
		}
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			double numerator =  /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];	// (19)'s numerator(분자)
			stats_matrix[r][c].weight = numerator / denominator; 

			if (stats_matrix[r][c].weight > 0.0) {
				if ((double)(stats_matrix[r][c].weight) < (double)(FLT_MIN) / (double)10.0) {
					std::cerr << "[" << iFrmCnt << "] weight(TW) < 0.1*FLT_MIN" << std::endl;
					m_cost[r*nStats + c] = 9999; // log(1.0)
				}
				else m_cost[r*nStats + c] = -100.0*log2l((double)numerator);			
			}
			else {
				m_cost[r*nStats + c] = 10000;
			}
		}
	}
	);

	int min_cost = INT_MAX, max_cost = 0;
	for (int r = 0; r < obss_live.size(); ++r) {
		for (int c = 0; c < mStats; ++c) {
			if (min_cost > m_cost[r*mStats + c]) min_cost = m_cost[r*mStats + c];
			if (max_cost < m_cost[r*mStats + c]) max_cost = m_cost[r*mStats + c];
		}
	}

	// Hungarian Method for solving data association problem (find the max cost assignment pairs)
	std::vector<vector<int>> assigns;
	assigns = this->HungarianMethod(m_cost, obss_live.size(), stats_lost.size(), max_cost);


	bool *isAssignedStats = new bool[stats_lost.size()];	memset(isAssignedStats, 0, stats_lost.size());
	bool *isAssignedObs = new bool[obss_live.size()];	memset(isAssignedObs, 0, obss_live.size());

	for (int r = 0; r < obss_live.size(); ++r) {
		obss_live[r].id_associated = -1; // faild to tracklet association
		for (int c = 0; c < stats_lost.size(); ++c) {
			if (assigns[r][c] == 1 && m_cost[r*stats_lost.size() + c] < max_cost) {

				// obss_live[r].id = stats_lost[c].id;
				obss_live[r].id_associated = stats_lost[c].id;
				obss_live[r].fn_latest_T2TA = iFrmCnt;

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;

				stats_lost[c].isAlive = true;

				break;
			}
		}
	}

	delete[]m_cost;
	delete[]isAssignedObs;
	delete[]isAssignedStats;
}
void GMPHD_OGM::CheckOcclusionsMergeStates(vector<BBTrk>& stats, const double T_merge, const double T_occ) {

	double MERGE_THRESHOLD = T_merge;
	double OCCLUSION_THRESHOLD = T_occ;
	if (!MERGE_ON)						MERGE_THRESHOLD = 1.0;		// do not merge any target
																	//if (!OCC_HANDLING_FRAME_WISE_ON)	OCCLUSION_THRESHOLD = 2.0;  // do not handle any occlusion

	vector<vector<RectID>> mergeStatsIdxes(stats.size());
	mergeStatsIdxes.resize(stats.size());
	vector<vector<double>> mergeStatsORs(stats.size()); // OR: Overlapping Ratio [0.0, 1.0]
	mergeStatsORs.resize(stats.size());

	vector<vector<bool>> visitTable;
	visitTable.resize(stats.size(), std::vector<bool>(stats.size(), false));

	int* mergeIdxTable = new int[stats.size()];

	double* overlapRatioTable = new double[stats.size()];

	// Init & clearing before checking occlusion
	for (int i = 0; i < stats.size(); ++i) {
		visitTable[i][i] = true;
		mergeIdxTable[i] = -1;
		overlapRatioTable[i] = 0.0;
		stats.at(i).isOcc = false;
		stats.at(i).occTargets.clear();
	}
	for (int a = 0; a < stats.size(); ++a) {


		if (stats.at(a).isAlive) {

			cv::Rect Ra = stats.at(a).rec;
			cv::Point Pa = cv::Point(Ra.x + Ra.width / 2, Ra.y + Ra.height / 2);

			for (int b = a + 1; b < stats.size(); ++b) {
				if (stats.at(b).isAlive && !visitTable[a][b] && !visitTable[b][a]) { // if a pair is not yet visited
					int min_id = stats.at(a).id;
					if (stats.at(b).id < stats.at(a).id) min_id = stats.at(b).id;

					cv::Rect Rb = stats.at(b).rec;
					cv::Point Pb = cv::Point(Rb.x + Rb.width / 2, Rb.y + Rb.height / 2);

					// check overlapping region
					double Ua = (double)(Ra & Rb).area() / (double)Ra.area();
					double Ub = (double)(Ra & Rb).area() / (double)Rb.area();

					double SIOA = (Ua + Ub) / 2.0; // Symmetric, The Sum-of-Intersection-over-Area (SIOA)
					double IOU = (double)(Ra & Rb).area() / (double)(Ra | Rb).area();

					// Size condition					
					if ((Ra.area() > (Rb.area() * SIZE_CONSTRAINT_RATIO)) || (Rb.area() > (Ra.area() * SIZE_CONSTRAINT_RATIO))) {
						SIOA = 0.0;
						IOU = 0.0;
					}

					double MERGE_MEASURE_VALUE = 0.0;

					if (this->params.MERGE_METRIC == MERGE_METRIC_SIOA)		MERGE_MEASURE_VALUE = SIOA;
					if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)		MERGE_MEASURE_VALUE = IOU;

					if (MERGE_MEASURE_VALUE >= MERGE_THRESHOLD) {

						stats.at(a).iGroupID = min_id;
						stats.at(b).iGroupID = min_id;

						mergeStatsIdxes[a].push_back(RectID(stats.at(b).id, stats.at(b).rec));
						mergeStatsIdxes[a].back().idx = b;
						mergeStatsIdxes[a].back().min_id = min_id;

						mergeStatsIdxes[b].push_back(RectID(stats.at(a).id, stats.at(a).rec));
						mergeStatsIdxes[b].back().idx = a;
						mergeStatsIdxes[b].back().min_id = min_id;

						mergeStatsORs[a].push_back(SIOA);
						mergeStatsORs[b].push_back(SIOA);

					}
					else if (MERGE_MEASURE_VALUE < MERGE_THRESHOLD && MERGE_MEASURE_VALUE>OCCLUSION_THRESHOLD) { // occlusion

						stats.at(a).isOcc = true;
						stats.at(b).isOcc = true;

						stats.at(a).occTargets.push_back(RectID(stats.at(b).id, stats.at(b).rec));
						stats.at(a).occTargets.back().min_id = min_id;
						stats.at(a).iGroupID = min_id;

						stats.at(b).occTargets.push_back(RectID(stats.at(a).id, stats.at(a).rec));
						stats.at(b).occTargets.back().min_id = min_id;
						stats.at(b).iGroupID = min_id;

					}

					// check visiting
					visitTable[a][b] = true;
					visitTable[b][a] = true;
				}
			}
			//cv::waitKey();

			// final check
			if (!stats.at(a).occTargets.empty()) {
				stats.at(a).isOcc = true;
			}
			else {
				stats.at(a).isOcc = false;
			}
		}
	}

	// Find Merge Group
	if (GROUP_MANAGEMENT_FRAME_WISE_ON) {
		for (int a = 0; a < stats.size(); ++a) {
			if (stats.at(a).isAlive && !mergeStatsIdxes[a].empty()) {

				int min_id = stats.at(a).id;
				for (int m = 0; m < mergeStatsIdxes[a].size(); m++) {
					if (min_id > stats[mergeStatsIdxes[a][m].idx].id)
						min_id = stats[mergeStatsIdxes[a][m].idx].id;
				}
				for (int m = 0; m < mergeStatsIdxes[a].size(); m++) {
					stats[mergeStatsIdxes[a][m].idx].iGroupID = min_id;
				}
				stats.at(a).iGroupID = min_id;
				stats.at(a).iGroupID = this->FindMinIDofNeigbors2Depth(stats, mergeStatsIdxes[a], stats.at(a).iGroupID);
			}
		}
		// Find Occlusion Group & Energy minimization (in low-level)
		for (int a = 0; a < stats.size(); ++a) {
			if (stats.at(a).isAlive /*&& !stats.at(a).occTargets.empty()*/) {

				stats.at(a).iGroupID = this->FindMinIDofNeigbors2Depth(stats, stats.at(a).occTargets, stats.at(a).iGroupID);
			}
		}
	}

	// Merge
	for (int a = 0; a < stats.size(); ++a) {
		if (stats.at(a).isAlive && !mergeStatsIdxes[a].empty()) {

			// Do not merge tracklet-level occludded (>= SIOA threshold) targets 
			if (GROUP_MANAGEMENT_FRAME_WISE_ON && !this->groupsBatch[this->params.GROUP_QUEUE_SIZE - 1][stats.at(a).iGroupID].empty()) {
				// Do nothing if the group id exists in the Group Queue
				stats.at(a).isMerged = false;
				stats.at(a).isAlive = true;
			}
			else {
				if (stats.at(a).id == stats.at(a).iGroupID) {

					for (int m = 0; m < mergeStatsIdxes[a].size(); m++) {
						stats.at(a).rec.x = 0.1*stats.at(mergeStatsIdxes[a][m].idx).rec.x + 0.9*stats.at(a).rec.x;
						stats.at(a).rec.y = 0.1*stats.at(mergeStatsIdxes[a][m].idx).rec.y + 0.9*stats.at(a).rec.y;
						stats.at(a).rec.width = 0.1*stats.at(mergeStatsIdxes[a][m].idx).rec.width + 0.9*stats.at(a).rec.width;
						stats.at(a).rec.height = 0.1*stats.at(mergeStatsIdxes[a][m].idx).rec.height + 0.9*stats.at(a).rec.height;
					}
					stats.at(a).isMerged = false;	// Other IDs are merged into stats[a]
					stats.at(a).isAlive = true;
				}
				else {
					stats.at(a).isMerged = true;	// stats[a] is merged into an oldest state having the smallest ID
					stats.at(a).isAlive = false;
				}
			}
		}
	}
	delete[]mergeIdxTable;
	delete[]overlapRatioTable;
}
int GMPHD_OGM::FindMinIDofNeigbors2Depth(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id) {
	int min_id = parent_occ_group_min_id;
	vector<RectID>::iterator iterR;
	for (iterR = occ_targets.begin(); iterR != occ_targets.end(); ++iterR) {
		vector<BBTrk>::iterator iterT;
		for (iterT = targets.begin(); iterT != targets.end(); ++iterT) {

			if ((iterR->id == iterT->id) && !iterT->occTargets.empty()) {
				if (min_id > iterT->occTargets[0].min_id) { // when mininmum IDs in occlusion group are different

					min_id = iterT->occTargets[0].min_id;
				}
			}
		}
	}
	return min_id;
}
void GMPHD_OGM::CheckOcclusionsGroups(vector<BBTrk>& stats, const double T_merge, const double T_occ) {

	double MERGE_THRESHOLD = T_merge;
	double OCCLUSION_THRESHOLD = T_occ;
	if (!MERGE_ON)						MERGE_THRESHOLD = 1.0;		// do not merge any target
																	//if (!OCC_HANDLING_FRAME_WISE_ON)	OCCLUSION_THRESHOLD = 2.0;  // do not handle any occlusion

	vector<vector<int>> mergeStatsIdxes(stats.size());
	mergeStatsIdxes.resize(stats.size());
	vector<vector<double>> mergeStatsORs(stats.size()); // OR: Overlapping Ratio [0.0, 2.0]
	mergeStatsORs.resize(stats.size());

	vector<vector<bool>> visitTable;
	visitTable.resize(stats.size(), std::vector<bool>(stats.size(), false));
	for (int v = 0; v < stats.size(); v++) visitTable[v][v] = true;

	//int* mergeIdxTable = new int[stats.size()];
	//for (int i = 0; i < stats.size(); i++) mergeIdxTable[i] = -1;

	//double* overlapRatioTable = new double[stats.size()];
	//for (int i = 0; i < stats.size(); i++) overlapRatioTable[i] = 0.0;

	// Init & clearing before checking occlusion
	for (int a = 0; a < stats.size(); ++a) {
		stats.at(a).isOcc = false;
		stats.at(a).occTargets.clear();
	}

	for (int a = 0; a < stats.size(); ++a) {

		if (stats.at(a).isAlive) {

			cv::Rect Ra = stats.at(a).rec;
			cv::Point Pa = cv::Point(Ra.x + Ra.width / 2, Ra.y + Ra.height / 2);

			for (int b = a + 1; b < stats.size(); ++b) {
				if (stats.at(b).isAlive && !visitTable[a][b] && !visitTable[b][a]) { // if a pair is not yet visited


					cv::Rect Rb = stats.at(b).rec;
					cv::Point Pb = cv::Point(Rb.x + Rb.width / 2, Rb.y + Rb.height / 2);

					// check overlapping region
					double Ua = (double)(Ra & Rb).area() / (double)Ra.area();
					double Ub = (double)(Ra & Rb).area() / (double)Rb.area();
					double SIOA = (Ua + Ub) / 2.0; // Symmetric

					double IOU = (double)(Ra & Rb).area() / (double)(Ra | Rb).area();

					// Size condition					
					if ((Ra.area() > (Rb.area()*(SIZE_CONSTRAINT_RATIO + 1))) || (Rb.area() > (Ra.area() * (SIZE_CONSTRAINT_RATIO + 1)))) {
						SIOA = 0.0;
						IOU = 0.0;
					}

					double MERGE_MEASURE_VALUE = 0.0;

					if (this->params.MERGE_METRIC == MERGE_METRIC_SIOA)		MERGE_MEASURE_VALUE = SIOA;
					if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)		MERGE_MEASURE_VALUE = IOU;

					if (MERGE_MEASURE_VALUE < MERGE_THRESHOLD && MERGE_MEASURE_VALUE>OCCLUSION_THRESHOLD) { // occlusion
						stats.at(a).isOcc = true;
						stats.at(b).isOcc = true;
						stats.at(a).occTargets.push_back(RectID(stats.at(b).id, stats.at(b).rec));
						stats.at(b).occTargets.push_back(RectID(stats.at(a).id, stats.at(a).rec));
					}

					// check visiting
					visitTable[a][b] = true;
					visitTable[b][a] = true;
				}
			}

			// final check
			if (!stats.at(a).occTargets.empty()) {
				stats.at(a).isOcc = true;
			}
			else {
				stats.at(a).isOcc = false;
			}
		}

		// Find the minimum id within an occlusion group (key for groups' map container)
		int min_id = stats.at(a).id;
		vector<RectID>::iterator iterR;
		for (iterR = stats.at(a).occTargets.begin(); iterR != stats.at(a).occTargets.end(); ++iterR) {
			if (min_id > iterR->id) min_id = iterR->id;
		}
		for (iterR = stats.at(a).occTargets.begin(); iterR != stats.at(a).occTargets.end(); ++iterR) {
			iterR->min_id = min_id;
		}
	}
}
void GMPHD_OGM::UnifyNeighborGroups(vector<BBTrk> input_targets) {

	map<int, vector<RectID>> groups;

	// Iterate the live objects
	vector<BBTrk>::iterator iterT;
	for (iterT = input_targets.begin(); iterT != input_targets.end(); ++iterT) {

		// Iterate the occluded objects vector of an object
		if (!iterT->occTargets.empty()) {

			int key_min_id = FindMinIDofNeigbors2Depth(input_targets, iterT->occTargets, iterT->occTargets[0].min_id);		// 2-Depth Search
			//int key_min_id = FindMinIDofNeigborsRecursive(input_targets, iterT->occTargets, iterT->occTargets[0].min_id);	// recursive function

			vector<RectID> occ_group;

			pair<map<int, vector<RectID>>::iterator, bool> isEmpty = groups.insert(map<int, vector<RectID>>::value_type(key_min_id, occ_group));

			if (isEmpty.second == false) { // already exists

				vector<RectID>::iterator iterR;
				for (iterR = iterT->occTargets.begin(); iterR != iterT->occTargets.end(); ++iterR)
				{
					bool isDuplicated = false;
					vector<RectID>::iterator iterG;
					for (iterG = groups[key_min_id].begin(); iterG != groups[key_min_id].end(); ++iterG) {
						if (iterG->id == iterR->id) {
							isDuplicated = true;
							break;
						}
					}
					if (!isDuplicated) {
						occ_group.push_back(iterR[0]);
					}
				}
				groups[key_min_id].insert(groups[key_min_id].end(), occ_group.begin(), occ_group.end());
			}
			else {							// newly added
				groups[key_min_id] = iterT->occTargets;
				vector<RectID>::iterator iterRtemp;
			}
		}
	}

	// Update the queue of the groups
	for (int g = 0; g < this->params.GROUP_QUEUE_SIZE - 1; ++g) {
		this->groupsBatch[g].clear();
		this->groupsBatch[g] = this->groupsBatch[g + 1];
	}
	this->groupsBatch[this->params.GROUP_QUEUE_SIZE - 1] = groups;
}
/**
*	@brief
*	@details
*	@param int iFrmCnt input
*	@param int group_min_id input
*	@param cv::Rect group_rect input
*	@param cv::Rect* objs output
*	@param vector<RectID> groupRects input
*	@param vector<BBTrk> liveReliables output
*
*	@return double the cost value between the objects in a group
*
*/
vector<double[2]> GMPHD_OGM::MinimizeGroupCost(int iFrmCnt, int group_min_id, cv::Rect group_rect, vector<RectID>* objs, vector<RectID> groupRects, vector<BBTrk>& liveReliables) {

	double group_costs[2] = { 0.5 ,0.5 }; // Cost of Neutral State
	int nSize = groupRects.size();
	vector<double[2]> costs_vec(nSize*(nSize - 1));

	std::sort(groupRects.begin(), groupRects.end()); // sort vector<RectID> rects in a group by their IDs (ascending order)

	// Build the Gaussian Mixture Model representing the relative motion between the objects in a group
													 
	cv::Mat cov = (cv::Mat_<double>(2, 2) << \
		VAR_X * 4, 0, \
		0, VAR_Y * 4);

	if (groupRects.size() == 2) {

		int id[2] = { groupRects[0].id,groupRects[1].id };
		double w[2] = { this->tracks_reliable[id[0]].back().weight , this->tracks_reliable[id[1]].back().weight };
		double sumWeight = w[0] + w[1];
		w[0] /= sumWeight; w[1] /= sumWeight;

		cv::Rect mean_rec[2];
		int SIZES[2];
		for (int p = 0; p < 2; p++)
			SIZES[p] = this->tracks_reliable[id[p]].size();

		objs[0].push_back(groupRects[0]);
		objs[0].push_back(groupRects[1]);
		objs[1].push_back(RectID(id[0], this->tracks_reliable[id[0]].back().rec_corr)); // predicted BB
		objs[1].push_back(RectID(id[1], this->tracks_reliable[id[1]].back().rec_corr));

		// Mean vectors of GMM
		cv::Mat m[2];
		this->CvtRec2Mat4TopologyModel(2, objs[1][0].rec, objs[1][1].rec, m[0]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][1].rec, objs[1][0].rec, m[1]);

		// Observation vectors of GMM (input)
		cv::Mat x[2];
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[0]].back().rec, this->tracks_reliable[id[1]].back().rec, x[0]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[1]].back().rec, this->tracks_reliable[id[0]].back().rec, x[1]);

		bool bIntrinsic_correction[2][2] = { { false,false },{ false,false } };
		double c[2][2];
		double lnc[2][2];
		int hypotheses[2][2] = \
		{ {0, 1},
		{ 1,0 }};
		int pIdices[2][2][2] = {
			{ { 0,0 },{ 1,1 } },
			{ { 1,0 },{ 0,1 } }
		};

		for (int p = 0; p < 2; ++p) {
			c[0][p] = this->GaussianFunc(2, x[p], m[p], cov);
			c[1][p] = this->GaussianFunc(2, x[(p + 1) % 2], m[p], cov);

			// L2-norm		
			for (int h = 0; h < 2; h++) {	// extrinsic hypothesis analysis
				if (c[h][p] <= Q_TH_LOW_15) {
					//w[p] = 0.5;
					c[h][p] = Q_TH_LOW_15;
					lnc[h][p] = -1000;
					// check intrinsic correction
					bIntrinsic_correction[h][p] = true;
				}
				else {
					lnc[h][p] = log2l(c[h][p]);
				}
			}

		}
		// L2-norm

		int hIdx_min = 0;
		double min_cost = DBL_MAX;
		bool bIC[2] = { false,false };
		for (int h = 0; h < 2; ++h) {
			double costs[2] = { -1.0 * (log2l(w[0]) + lnc[h][0] + log2l(w[1]) + lnc[h][1]), 10000 };
			//printf("[Hypothesis %d]\n",h+1);
			for (int p = 0; p < 2; ++p) {


				if (bIntrinsic_correction[h][p]) {
					bIC[h] = true;
					c[h][p] = this->GaussianFunc(2, m[pIdices[h][p][1]], m[pIdices[h][p][1]], cov);
					lnc[h][p] = log2l(c[h][p]);
					if (c[h][p] <= Q_TH_LOW_15) {
						c[h][p] = Q_TH_LOW_15;
						lnc[h][p] = -1000;
					}
				}
			}
			if (bIC[h]) costs[1] = -1.0 * (log2l(w[0]) + lnc[h][0] + log2l(w[1]) + lnc[h][1]);
			costs_vec[h][0] = costs[0];
			costs_vec[h][1] = costs[1];
			if (min_cost < costs_vec[h][0]) {
				min_cost = costs_vec[h][0];
				hIdx_min = h;
			}
		}
		// intrinsic correction of an optimal hypothesis
		int idices_latency[2], idices[2];
		cv::Rect rects_copy[2];
		for (int i = 0; i < liveReliables.size(); ++i) {
			if (liveReliables[i].id == groupRects[0].id) idices_latency[0] = i;
			if (liveReliables[i].id == groupRects[1].id) idices_latency[1] = i;
		}
		for (int i = 0; i < this->liveTrkVec.size(); ++i) {
			if (this->liveTrkVec[i].id == groupRects[0].id) idices[0] = i;
			if (this->liveTrkVec[i].id == groupRects[1].id) idices[1] = i;
		}
		for (int v = 0; v < 2; ++v) {
			rects_copy[v] = liveReliables[idices_latency[hypotheses[hIdx_min][v]]].rec;

			if (bIC[hIdx_min]) {
				rects_copy[v] = objs[1][hypotheses[hIdx_min][v]].rec;
			}
		}
		for (int v = 0; v < 2; ++v) {

			liveReliables[idices_latency[v]].rec = this->cvMergeRects(liveReliables[idices_latency[v]].rec, rects_copy[v], 0.5);

			this->tracks_reliable[groupRects[v].id].back().rec = this->cvMergeRects(this->tracks_reliable[groupRects[v].id].back().rec, rects_copy[v], 0.5);

			vector<BBTrk> track;
			int tSize = this->tracksbyID[groupRects[v].id].size();
			for (int fr = 0;; fr++) {
				this->tracksbyID[groupRects[v].id][tSize - 1 - fr].id = groupRects[hypotheses[hIdx_min][v]].id;
				track.push_back(this->tracksbyID[groupRects[v].id][tSize - 1 - fr]);
				this->tracksbyID[groupRects[v].id].pop_back();
				if (this->tracksbyID[groupRects[v].id][tSize - 1 - fr].fn == sysFrmCnt - this->params.FRAMES_DELAY_SIZE) {
					this->tracksbyID[groupRects[v].id][tSize - 1 - fr].rec = this->cvMergeRects(this->tracksbyID[groupRects[v].id][tSize - 1 - fr].rec, rects_copy[v], 0.5);
					break;
				}
			}
			this->tracksbyID[groupRects[hypotheses[hIdx_min][v]].id].insert(this->tracksbyID[groupRects[hypotheses[hIdx_min][v]].id].end(), track.begin(), track.end());
			this->liveTrkVec[idices[v]].id = groupRects[hypotheses[hIdx_min][v]].id;
			this->liveTracksBatch[this->params.FRAMES_DELAY_SIZE][idices[v]].id = groupRects[hypotheses[hIdx_min][v]].id;
		}

		return costs_vec; // scale 값이 필요하겠군.. alpha
	}
	if (groupRects.size() == 3) {

		int id[3] = { groupRects[0].id,groupRects[1].id,groupRects[2].id };
		double w[6] = { this->tracks_reliable[id[0]].back().weight , this->tracks_reliable[id[0]].back().weight , this->tracks_reliable[id[1]].back().weight,\
			this->tracks_reliable[id[1]].back().weight, this->tracks_reliable[id[2]].back().weight, this->tracks_reliable[id[2]].back().weight };
		double sumWeight = w[0] + w[1] + w[2] + w[3] + w[4] + w[5];
		w[0] /= sumWeight; w[1] /= sumWeight; w[2] /= sumWeight;
		w[3] /= sumWeight; w[4] /= sumWeight; w[5] /= sumWeight;

		cv::Rect mean_rec[3];
		int SIZES[3];
		for (int p = 0; p < 3; p++)
			SIZES[p] = this->tracks_reliable[id[p]].size();

		objs[0].push_back(groupRects[0]); objs[0].push_back(groupRects[1]); objs[0].push_back(groupRects[2]);
		objs[1].push_back(RectID(id[0], this->tracks_reliable[id[0]].back().rec_corr));
		objs[1].push_back(RectID(id[1], this->tracks_reliable[id[1]].back().rec_corr));
		objs[1].push_back(RectID(id[2], this->tracks_reliable[id[2]].back().rec_corr));


		// Mean vectors of GMM
		/// 0=2, 1=4, 2=3
		cv::Mat m[6];
		this->CvtRec2Mat4TopologyModel(2, objs[1][0].rec, objs[1][1].rec, m[0]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][0].rec, objs[1][2].rec, m[1]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][1].rec, objs[1][0].rec, m[2]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][1].rec, objs[1][2].rec, m[3]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][2].rec, objs[1][0].rec, m[4]);
		this->CvtRec2Mat4TopologyModel(2, objs[1][2].rec, objs[1][1].rec, m[5]);

		// Observation vectors of GMM (input)
		/// 0=2, 1=4, 3=5
		cv::Mat x[6];
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[0]].back().rec, this->tracks_reliable[id[1]].back().rec, x[0]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[0]].back().rec, this->tracks_reliable[id[2]].back().rec, x[1]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[1]].back().rec, this->tracks_reliable[id[0]].back().rec, x[2]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[1]].back().rec, this->tracks_reliable[id[2]].back().rec, x[3]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[2]].back().rec, this->tracks_reliable[id[0]].back().rec, x[4]);
		this->CvtRec2Mat4TopologyModel(2, this->tracks_reliable[id[2]].back().rec, this->tracks_reliable[id[1]].back().rec, x[5]);

		bool bIntrinsic_correction[6][6] = \
		{ { false, false, false, false, false, false }, \
		{ false, false, false, false, false, false },
		{ false,false,false,false,false,false },
		{ false,false,false,false,false,false },
		{ false,false,false,false,false,false },
		{ false,false,false,false,false,false }};
		double c[6][6];
		double lnc[6][6];
		int hypotheses[6][3] = \
		{ {0, 1, 2},
		{ 0,2,1 },
		{ 1,0,2 },
		{ 1,2,0 },
		{ 2,0,1 },
		{ 2,1,0 }};
		int pIdices[6][6][2] = \
		{ { {0, 0}, { 1,1 }, { 2,2 }, { 3,3 }, { 4,4 }, { 5,5 } }, \
		{ {1, 0}, { 0,1 }, { 4,2 }, { 5,3 }, { 2,4 }, { 3,5 } },
		{ { 2,0 },{ 3,1 },{ 0,2 },{ 1,3 },{ 5,4 },{ 4,5 } },
		{ { 3,0 },{ 2,1 },{ 5,2 },{ 4,3 },{ 0,4 },{ 1,5 } },
		{ { 4,0 },{ 5,1 },{ 1,2 },{ 0,3 },{ 3,4 },{ 2,5 } },
		{ { 5,0 },{ 4,1 },{ 3,2 },{ 2,3 },{ 1,4 },{ 0,5 } } };
		//printf("[ID%d,ID%d,ID%d]:%lf\n", id[0], id[1], id[2], sumWeight);

		concurrency::parallel_for(0, 6, [&](int h) {

			for (int p = 0; p < 6; ++p) { // extrinsic hypothesis analysis
				c[h][p] = this->GaussianFunc(2, x[pIdices[h][p][0]], m[pIdices[h][p][1]], cov);

				// L2-norm
				if (c[h][p] <= Q_TH_LOW_15) {
					//w[p] = 1.0/3.0;
					c[h][p] = Q_TH_LOW_15; // ln(c[p]) = -34.53
					lnc[h][p] = -1000;
					// check intrinsic correction
					bIntrinsic_correction[h][p] = true;
				}
				else {
					lnc[h][p] = log2l(c[h][p]);
				}
			}

		});

		// L2-norm
		int hIdx_min = 0;
		double min_cost = DBL_MAX;
		bool bIC[6] = { false,false,false,false,false,false };
		for (int h = 0; h < 6; ++h) {
			double costs[2] = { -1.0 * (log2l(w[0] * w[1] * w[2] * w[3] * w[4] * w[5]) + lnc[h][0] + lnc[h][1] + lnc[h][2] + lnc[h][3] + lnc[h][4] + lnc[h][5]), 10000 };

			//printf("[Hypothesis %d]\n", h);
			for (int p = 0; p < 6; ++p) {

				if (bIntrinsic_correction[h][p]) {
					bIC[h] = true;
					c[h][p] = this->GaussianFunc(2, m[pIdices[h][p][1]], m[pIdices[h][p][1]], cov);
					lnc[h][p] = log2l(c[h][p]);
					if (c[h][p] <= Q_TH_LOW_15) {
						c[h][p] = Q_TH_LOW_15;
						lnc[h][p] = -1000;
					}
				}
			}
			if (bIC[h]) costs[1] = -1.0 * (log2l(w[0] * w[1] * w[2] * w[3] * w[4] * w[5]) + lnc[h][0] + lnc[h][1] + lnc[h][2] + lnc[h][3] + lnc[h][4] + lnc[h][5]);
			costs_vec[h][0] = costs[0];
			costs_vec[h][1] = costs[1];
			if (min_cost < costs_vec[h][0]) {
				min_cost = costs_vec[h][0];
				hIdx_min = h;
			}
		}

		// intrinsic correction of an optimal hypothesis  
		int idices_latency[3], idices[3];
		cv::Rect rects_copy[3];
		for (int i = 0; i < liveReliables.size(); ++i) {
			if (liveReliables[i].id == groupRects[0].id) idices_latency[0] = i;
			if (liveReliables[i].id == groupRects[1].id) idices_latency[1] = i;
			if (liveReliables[i].id == groupRects[2].id) idices_latency[2] = i;
		}
		for (int i = 0; i < this->liveTrkVec.size(); ++i) {
			if (this->liveTrkVec[i].id == groupRects[0].id) idices[0] = i;
			if (this->liveTrkVec[i].id == groupRects[1].id) idices[1] = i;
			if (this->liveTrkVec[i].id == groupRects[2].id) idices[2] = i;
		}
		for (int v = 0; v < 3; ++v) {
			rects_copy[v] = liveReliables[idices_latency[hypotheses[hIdx_min][v]]].rec;

			if (bIC[hIdx_min]) {
				rects_copy[v] = objs[1][hypotheses[hIdx_min][v]].rec;
			}
		}
		for (int v = 0; v < 3; ++v) {

			liveReliables[idices_latency[v]].rec = this->cvMergeRects(liveReliables[idices_latency[v]].rec, rects_copy[v], 0.5);
			this->tracks_reliable[groupRects[v].id].back().rec = this->cvMergeRects(this->tracks_reliable[groupRects[v].id].back().rec, rects_copy[v], 0.5);

			vector<BBTrk> track;
			int tSize = this->tracksbyID[groupRects[v].id].size();
			for (int fr = 0;; fr++) {
				this->tracksbyID[groupRects[v].id][tSize - 1 - fr].id = groupRects[hypotheses[hIdx_min][v]].id;
				track.push_back(this->tracksbyID[groupRects[v].id][tSize - 1 - fr]);
				this->tracksbyID[groupRects[v].id].pop_back();
				if (this->tracksbyID[groupRects[v].id][tSize - 1 - fr].fn == sysFrmCnt - this->params.FRAMES_DELAY_SIZE) {
					this->tracksbyID[groupRects[v].id][tSize - 1 - fr].rec = this->cvMergeRects(this->tracksbyID[groupRects[v].id][tSize - 1 - fr].rec, rects_copy[v], 0.5);
					break;
				}
			}
			this->tracksbyID[groupRects[hypotheses[hIdx_min][v]].id].insert(this->tracksbyID[groupRects[hypotheses[hIdx_min][v]].id].end(), track.begin(), track.end());
			this->liveTrkVec[idices[v]].id = groupRects[hypotheses[hIdx_min][v]].id;
			this->liveTracksBatch[this->params.FRAMES_DELAY_SIZE][idices[v]].id = groupRects[hypotheses[hIdx_min][v]].id;
		}
		return costs_vec;
	}

	for (int i = 0; i < costs_vec.size(); ++i) {
		costs_vec[i][0] = group_costs[0];
		costs_vec[i][1] = group_costs[1];
	}

	return costs_vec;
}
cv::Rect GMPHD_OGM::CalGroupRect(int group_min_id, vector<RectID> groupRects) {

	cv::Rect group_rect = groupRects.front().rec;
	vector<RectID>::iterator iterR;
	for (iterR = groupRects.begin(); iterR != groupRects.end(); ++iterR) {
		group_rect = group_rect | iterR->rec;
	}
	return group_rect;
}
void GMPHD_OGM::CvtRec2Mat4TopologyModel(int dims, cv::Rect org, cv::Rect trg, cv::Mat& m) {

	m = cv::Mat(dims, 1, CV_64FC1);
	m = (cv::Mat_<double>(dims, 1) << trg.x + (double)trg.width / 2.0 - org.x - (double)org.width / 2.0, trg.y + (double)trg.height / 2.0 - org.y - (double)org.height / 2.0);
}
inline cv::Rect GMPHD_OGM::cvMergeRects(const cv::Rect rect1, const cv::Rect rect2, double alpha) {
	cv::Rect res;
	res.x = alpha*rect1.x + (1.0 - alpha)*rect2.x;
	res.y = alpha*rect1.y + (1.0 - alpha)*rect2.y;
	res.width = alpha*rect1.width + (1.0 - alpha)*rect2.width;
	res.height = alpha*rect1.height + (1.0 - alpha)*rect2.height;
	return res;
}
float GMPHD_OGM::CalcIOU(cv::Rect Ra, cv::Rect Rb) {
	// check overlapping region
	float Ua = (float)(Ra & Rb).area() / (float)Ra.area();
	float Ub = (float)(Ra & Rb).area() / (float)Rb.area();

	float IOU = (float)(Ra & Rb).area() / (float)(Ra | Rb).area(); // Intersection-over-Union (IOU)

	return IOU;
}
float GMPHD_OGM::CalcSIOA(cv::Rect Ra, cv::Rect Rb) {
	// check overlapping region
	float Ua = (float)(Ra & Rb).area() / (float)Ra.area();
	float Ub = (float)(Ra & Rb).area() / (float)Rb.area();

	float SIOA = (Ua + Ub) / 2.0; // Symmetric, The Sum-of-Intersection-over-Area (SIOA)

	return SIOA;
}
void GMPHD_OGM::SetParams(GMPHDOGMparams params) {
	this->params = params;
	this->InitializeTrackletsContainters();
}
void GMPHD_OGM::ArrangeTargetsVecsBatchesLiveLost() {
	vector<BBTrk> liveTargets;
	vector<BBTrk> lostTargets;
	for (int tr = 0; tr < this->liveTrkVec.size(); ++tr) {
		if (this->liveTrkVec[tr].isAlive) {
			liveTargets.push_back(this->liveTrkVec[tr]);
		}
		else if (!this->liveTrkVec[tr].isAlive && !this->liveTrkVec[tr].isMerged) {

			lostTargets.push_back(this->liveTrkVec[tr]);
		}
		else {
			// abandon the merged targets (When target a'ID and b'TD are merged with a'ID < b'TD, target b is abandoned and not considered as LB_ASSOCIATION) 
		}
	}
	this->liveTrkVec.swap(liveTargets);	// swapping the alive targets
	this->lostTrkVec.swap(lostTargets);	// swapping the loss tragets	
	liveTargets.clear();
	lostTargets.clear();
}
void GMPHD_OGM::PushTargetsVecs2BatchesLiveLost() {
	if (this->sysFrmCnt >= this->params.TRACK_MIN_SIZE) {
		for (int q = 0; q < this->params.FRAMES_DELAY_SIZE; q++) {
			for (int i = 0; i < this->liveTracksBatch[q].size(); i++)this->liveTracksBatch[q].at(i).Destroy();
			this->liveTracksBatch[q].clear();
			this->liveTracksBatch[q] = liveTracksBatch[q + 1];

			for (int i = 0; i < this->lostTracksBatch[q].size(); i++)this->lostTracksBatch[q].at(i).Destroy();
			this->lostTracksBatch[q].clear();
			this->lostTracksBatch[q] = lostTracksBatch[q + 1];
		}
		this->liveTracksBatch[this->params.FRAMES_DELAY_SIZE] = this->liveTrkVec;
		this->lostTracksBatch[this->params.FRAMES_DELAY_SIZE] = this->lostTrkVec;
	}
	else if (this->sysFrmCnt < this->params.TRACK_MIN_SIZE) {
		this->liveTracksBatch[this->sysFrmCnt] = this->liveTrkVec;
		this->lostTracksBatch[this->sysFrmCnt] = this->lostTrkVec;
	}
}
void GMPHD_OGM::SortTrackletsbyID(map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& targets) {
	pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty;
	for (int j = 0; j < targets.size(); j++)
	{
		// ArrangeTargetsVecsBatchesLiveLost 에 의해 alive track 들만 모여있는 상태
		int id = targets.at(j).id;

		// targets.at(j).fn = this->sysFrmCnt; // 이게 왜 안넘어 갔는지 미스테리다, 와.. prediction 에서 framenumber를 update 안해줬네..

		vector<BBTrk> tracklet;
		tracklet.push_back(targets.at(j));

		pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = tracksbyID.insert(map<int, vector<BBTrk>>::value_type(id, tracklet));

		if (isEmpty.second == false) { // already has a element with target.at(j).id
			tracksbyID[id].push_back(targets.at(j));
			//if (DEBUG_PRINT)
			//printf("[%d-%d]ID%d is updated into tracksbyID\n", this->sysFrmCnt, targets.at(j).fn, id);
		}
		else {
			//if (DEBUG_PRINT)
			//printf("[%d-%d]ID%d is newly added into tracksbyID\n",this->sysFrmCnt, targets.at(j).fn, id);
		}

	}
}
void GMPHD_OGM::ClassifyTrackletReliability(int iFrmCnt, map<int, vector<BBTrk>>& tracksbyID, map<int, vector<BBTrk>>& reliables, map<int, std::vector<BBTrk>>& unreliables) {

	map<int, vector<BBTrk>>::iterator iterID;

	for (iterID = tracksbyID.begin(); iterID != tracksbyID.end(); iterID++) {
		if (!iterID->second.empty()) {

			if (iterID->second.back().fn == iFrmCnt) {

				vector<BBTrk> tracklet;
				vector<BBTrk>::reverse_iterator rIterT;
				bool isFound = false;
				for (rIterT = iterID->second.rbegin(); rIterT != iterID->second.rend(); rIterT++) {
					if (rIterT->fn == iFrmCnt - this->params.FRAMES_DELAY_SIZE) {

						tracklet.push_back(rIterT[0]);
						isFound = true;
						break;
					}
				}
				if (isFound /*iterID->second.back().fn - iterID->second.front().fn >= FRAMES_DELAY*/) { // reliable (with latency)
					pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = reliables.insert(map<int, vector<BBTrk>>::value_type(iterID->first, tracklet));
					if (isEmpty.second == false) {
						reliables[iterID->first].push_back(tracklet[0]);
					}

					unreliables[iterID->first].clear();
				}
				else {																					// unreliable (witout latency)
					pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = unreliables.insert(map<int, vector<BBTrk>>::value_type(iterID->first, iterID->second));
					if (isEmpty.second == false)
						unreliables[iterID->first].push_back(iterID->second.back());
				}
			}

		}

	}
}
void GMPHD_OGM::ClassifyReliableTracklets2LiveLost(int iFrmCnt, const map<int, vector<BBTrk>>& reliables, vector<BBTrk>& liveReliables, vector<BBTrk>& lostReliables, vector<BBTrk>& obss) {

	map<int, vector<BBTrk>>::const_iterator iterT;
	for (iterT = reliables.begin(); iterT != reliables.end(); iterT++) {
		if (!iterT->second.empty()) {
			if (iterT->second.back().fn == iFrmCnt - this->params.FRAMES_DELAY_SIZE) {

				//if (this->params.TRACK_MIN_SIZE == 2) { 
				//if (iterT->second.size() == 1) {
				//	obss.push_back(iterT->second.back());
				//}
				//else if (iterT->second.size() > 1) {
				//	liveReliables.push_back(iterT->second.back());
				//}
				////}
				////else {	// Other Scenes for DPM
				////if (iterT->second.size() >= this->params.TRACK_MIN_SIZE && iterT->second.size() <= this->params.T2TA_MAX_INTERVAL) {
				////	obss.push_back(iterT->second.back());
				////}
				////else if (iterT->second.size() > this->params.T2TA_MAX_INTERVAL) {
				////liveReliables.push_back(iterT->second.back());
				////}
				////}
				liveReliables.push_back(iterT->second.back());

			}
			else if (iterT->second.back().fn < iFrmCnt - this->params.FRAMES_DELAY_SIZE) {
				//if(iterT->second.size()>1)
				lostReliables.push_back(iterT->second.back());
			}
		}
	}
}
void GMPHD_OGM::ArrangeRevivedTracklets(map<int, vector<BBTrk>>& tracks, vector<BBTrk>& lives) {

	// ID Management
	vector<BBTrk>::iterator iterT;
	for (iterT = lives.begin(); iterT != lives.end(); ++iterT) {
		if (iterT->id_associated >= 0) { // id != -1, succeed in ID recovery;

										 // input parameter 1: tracks
			int size_old = tracks[iterT->id_associated].size();
			tracks[iterT->id_associated].insert(tracks[iterT->id_associated].end(), tracks[iterT->id].begin(), tracks[iterT->id].end());
			int size_new = tracks[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i) {  // 뒤에 새로 붙은것의 ID를 복원시켜줌(associated 된 lostTrk의 것으로)
				tracks[iterT->id_associated].at(i).id = iterT->id_associated;
				tracks[iterT->id_associated].at(i).id_associated = iterT->id_associated;
			}
			tracks[iterT->id].clear();

			// this->tracksbyID
			size_old = this->tracksbyID[iterT->id_associated].size();
			this->tracksbyID[iterT->id_associated].insert(this->tracksbyID[iterT->id_associated].end(), this->tracksbyID[iterT->id].begin(), this->tracksbyID[iterT->id].end());
			size_new = this->tracksbyID[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i) {
				this->tracksbyID[iterT->id_associated].at(i).id = iterT->id_associated;
				this->tracksbyID[iterT->id_associated].at(i).id_associated = iterT->id_associated;
			}
			this->tracksbyID[iterT->id].clear();

			// this->liveTrkVec (no letancy tracking)
			vector<BBTrk>::iterator iterTfw; // frame-wise (no latency)
			for (iterTfw = this->liveTrkVec.begin(); iterTfw != this->liveTrkVec.end(); ++iterTfw) {
				if (iterTfw->id == iterT->id) {
					iterTfw->id = iterT->id_associated;
					iterTfw->id_associated = iterT->id_associated;
					break;
				}
			}
			// this->liveTracksBatch (at t-2, t-1, t)
			for (int b = 0; b < this->params.TRACK_MIN_SIZE; ++b) {
				for (iterTfw = this->liveTracksBatch[b].begin(); iterTfw != this->liveTracksBatch[b].end(); ++iterTfw) {
					if (iterTfw->id == iterT->id) {
						iterTfw->id = iterT->id_associated;
						iterTfw->id_associated = iterT->id_associated;
						break;
					}
				}
			}

			// input parameter 2: lives
			iterT->id = iterT->id_associated;
		}
	}
}
void GMPHD_OGM::SetTotalFrames(int nFrames) {
	this->iTotalFrames = nFrames;
}
GMPHDOGMparams GMPHD_OGM::GetParams() {
	return this->params;
}
// Initialized Images Queue
void GMPHD_OGM::InitializeImagesQueue(int width, int height) {
	this->imgBatch = new cv::Mat[this->params.TRACK_MIN_SIZE];
	for (int i = 0; i < this->params.TRACK_MIN_SIZE; i++) {
		this->imgBatch[i].release();
		this->imgBatch[i] = cv::Mat(height, width, CV_8UC3);
	}
	this->frmWidth = width;
	this->frmHeight = height;
}
// Initialized the STL Containters for GMPHDOGM
void GMPHD_OGM::InitializeTrackletsContainters() {
	this->detsBatch = new std::vector<BBDet>[this->params.TRACK_MIN_SIZE];
	this->liveTracksBatch = new std::vector<BBTrk>[this->params.TRACK_MIN_SIZE];
	this->lostTracksBatch = new std::vector<BBTrk>[this->params.TRACK_MIN_SIZE];
	this->groupsBatch = new std::map<int, std::vector<RectID>>[this->params.GROUP_QUEUE_SIZE];
}
// Initialize Color Tab
void GMPHD_OGM::InitializeColorTab()
{
	int a;
	for (a = 1; a*a*a < MAX_OBJECTS; a++);
	int n = 255 / (a - 1);
	IplImage *temp = cvCreateImage(cvSize(40 * (MAX_OBJECTS), 32), IPL_DEPTH_8U, 3);
	cvSet(temp, CV_RGB(0, 0, 0));
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < a; j++) {
			for (int k = 0; k < a; k++) {
				//if(i*a*a+j*a+k>MAX_OBJECTS) break;
				//printf("%d:(%d,%d,%d)\n",i*a*a +j*a+k,i*n,j*n,k*n);
				if (i*a*a + j*a + k == MAX_OBJECTS) break;
				color_tab[i*a*a + j*a + k] = CV_RGB(i*n, j*n, k*n);
				cvLine(temp, cvPoint((i*a*a + j*a + k) * 40 + 20, 0), cvPoint((i*a*a + j*a + k) * 40 + 20, 32), CV_RGB(i*n, j*n, k*n), 32);
			}
		}
	}
	//cvShowImage("(private)Color tap", temp);
	cvWaitKey(1);
	cvReleaseImage(&temp);
}
void GMPHD_OGM::InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, int dims_state, int dims_obs)
{
	/* Initialize the transition matrix F, from state_t-1 to state_t

	1	0  △t	0	0	0
	0	1	0  △t	0	0
	0	0	1	0	0	0
	0	0	0	1	0	0
	0	0	0	0	1	0
	0	0	0	0	0	1

	△t = 구현시에는 △frame으로 즉 1이다.
	*/
	F = cv::Mat::eye(dims_state, dims_state, CV_64FC1); // identity matrix
	F.at<double>(0, 2) = 1.0;///30.0; // 30fps라 가정, 나중에 계산할때 St = St-1 + Vt-1△t (S : location) 에서 
	F.at<double>(1, 3) = 1.0;///30.0; // Vt-1△t 의해 1/30 은 사라진다. Vt-1 (1frame당 이동픽셀 / 0.0333..), △t = 0.0333...

	if (dims_state == DIMS_STATE) {
		Q = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0.0, 0, 0, 0, \
			0, VAR_Y, 0, 0.0, 0, 0, \
			0.0, 0, VAR_X_VEL, 0, 0, 0, \
			0, 0.0, 0, VAR_Y_VEL, 0, 0, \
			0, 0, 0, 0, 0, 0, \
			0, 0, 0, 0, 0, 0);
		Q = 0.5 * Q;

		Ps = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, 0, 0, \
			0, VAR_Y, 0, 0, 0, 0, \
			0, 0, VAR_X_VEL, 0, 0, 0, \
			0, 0, 0, VAR_Y_VEL, 0, 0, \
			0, 0, 0, 0, VAR_WIDTH, 0, \
			0, 0, 0, 0, 0, VAR_HEIGHT);

		R = (cv::Mat_<double>(dims_obs, dims_obs) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);
		/*	Initialize the transition matrix H, transing the state_t to the observation_t(measurement) */
		H = (cv::Mat_<double>(dims_obs, dims_state) << \
			1, 0, 0, 0, 0, 0, \
			0, 1, 0, 0, 0, 0, \
			0, 0, 0, 0, 1, 0, \
			0, 0, 0, 0, 0, 1);
	}
	else if (dims_state == DIMS_STATE_MID) {
		Q = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);
		Q = 0.5 * Q;

		Ps = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);

		R = (cv::Mat_<double>(dims_obs, dims_obs) << \
			VAR_X, 0, \
			0, VAR_Y);

		/*	Initialize the transition matrix H, transing the state_t to the observation_t(measurement) */
		H = (cv::Mat_<double>(dims_obs, dims_state) << \
			1, 0, 0, 0, \
			0, 1, 0, 0);
	}
}
vector<BBDet> GMPHD_OGM::DetectionFilteringUsingConfidenceArea(vector<BBDet>& obss, const double T_merge, const double T_occ) {
	vector<BBDet> filteredDetVec;
	vector<BBTrk> stats;
	for (int m = 0; m < obss.size(); ++m) {
		BBTrk tr;
		tr.isAlive = true;
		tr.id = m;
		tr.rec = obss[m].rec;
		tr.weight = obss[m].confidence;
		tr.fn = obss[m].fn;
		stats.push_back(tr);
	}
	double MERGE_THRESHOLD = T_merge;
	double OCCLUSION_THRESHOLD = T_occ;
	if (!MERGE_ON)	MERGE_THRESHOLD = 1.0;	// do not merge any target

	vector<vector<RectID>> mergeStatsIdxes(stats.size());
	mergeStatsIdxes.resize(stats.size());
	vector<vector<double>> mergeStatsORs(stats.size()); // OR: Overlapping Ratio [0.0, 1.0]
	mergeStatsORs.resize(stats.size());

	vector<vector<bool>> visitTable;
	visitTable.resize(stats.size(), std::vector<bool>(stats.size(), false));

	int* mergeIdxTable = new int[stats.size()];

	double* overlapRatioTable = new double[stats.size()];

	// Init & clearing before checking occlusion
	for (int i = 0; i < stats.size(); ++i) {
		visitTable[i][i] = true;
		mergeIdxTable[i] = -1;
		overlapRatioTable[i] = 0.0;
		stats.at(i).isOcc = false;
		stats.at(i).occTargets.clear();
	}
	for (int a = 0; a < stats.size(); ++a) {

		if (stats.at(a).isAlive) {

			cv::Rect Ra = stats.at(a).rec;
			cv::Point Pa = cv::Point(Ra.x + Ra.width / 2, Ra.y + Ra.height / 2);
			//cv::rectangle(distImg, Ra, cv::Scalar(255, 255, 255), 2);

			for (int b = a + 1; b < stats.size(); ++b) {
				if (stats.at(b).isAlive && !visitTable[a][b] && !visitTable[b][a]) { // if a pair is not yet visited

					cv::Rect Rb = stats.at(b).rec;
					cv::Point Pb = cv::Point(Rb.x + Rb.width / 2, Rb.y + Rb.height / 2);

					// check overlapping region
					double Ua = (double)(Ra & Rb).area() / (double)Ra.area();
					double Ub = (double)(Ra & Rb).area() / (double)Rb.area();

					// To compare detection confidence values
					double Ca = stats.at(a).weight;
					double Cb = stats.at(b).weight;

					double SIOA = (Ua + Ub) / 2.0; // Symmetric, The Sum-of-Intersection-over-Area (SIOA)
					double IOU = (double)(Ra & Rb).area() / (double)(Ra | Rb).area();

					double MERGE_MEASURE_VALUE = 0.0;

					if (this->params.MERGE_METRIC == MERGE_METRIC_SIOA)		MERGE_MEASURE_VALUE = SIOA;
					if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)		MERGE_MEASURE_VALUE = IOU;


					//char carrDist[10]; sprintf(carrDist,"%.lf",dist);
					//cv::putText(distImg, string(carrDist), cv::Point((Pa.x + Pb.x) / 2, (Pa.y + Pb.y) / 2), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
					if (MERGE_MEASURE_VALUE >= MERGE_THRESHOLD) {


						if (((Ra & Rb).area() >= 0.85 * Ra.area()) /*&& (Ra.width >= 0.6*Rb.width)*/) { // Ra in Rb (Rb > Ra)
							if (Ca >= 2.0*Cb) { // Filtering
								stats.at(b).isAlive = false;
							}
							if (Cb >= 2.0*Ca) { // Merge
								double Csum = Cb + Ca;
								double Wa = Ca / Csum;
								double Wb = Cb / Csum;
								stats.at(a).rec.x = Wb*stats.at(b).rec.x + Wa*stats.at(a).rec.x;
								stats.at(a).rec.y = Wb*stats.at(b).rec.y + Wa*stats.at(a).rec.y;
								stats.at(a).rec.width = Wb*stats.at(b).rec.width + Wa*stats.at(a).rec.width;
								stats.at(a).rec.height = Wb*stats.at(b).rec.height + Wa*stats.at(a).rec.height;
							}
						}
						if (((Ra & Rb).area() >= 0.85 * Rb.area()) /*&& (Ra.width >= 0.6*Rb.width)*/) { // Rb in Ra (Ra > Rb)
							if (Ca >= 2.0*Cb) { // Merge
								double Csum = Cb + Ca;
								double Wa = Ca / Csum;
								double Wb = Cb / Csum;
								stats.at(b).rec.x = Wb*stats.at(b).rec.x + Wa*stats.at(a).rec.x;
								stats.at(b).rec.y = Wb*stats.at(b).rec.y + Wa*stats.at(a).rec.y;
								stats.at(b).rec.width = Wb*stats.at(b).rec.width + Wa*stats.at(a).rec.width;
								stats.at(b).rec.height = Wb*stats.at(b).rec.height + Wa*stats.at(a).rec.height;
							}
							if (Cb >= 2.0*Ca) { // Filtering
								stats.at(a).isAlive = false;
							}
						}
					}

					// check visiting
					visitTable[a][b] = true;
					visitTable[b][a] = true;
				}
			}
		}
	}

	for (int n = 0; n < stats.size(); ++n) {
		if (stats.at(n).isAlive) {
			BBDet det;
			det.rec = stats[n].rec;
			det.confidence = stats[n].weight;
			det.fn = stats[n].fn;
			filteredDetVec.push_back(det);
		}
	}

	delete[]mergeIdxTable;
	delete[]overlapRatioTable;

	return filteredDetVec;
}
bool GMPHD_OGM::IsOutOfFrame(cv::Rect obj_rec, int fWidth, int fHeight) {
	cv::Rect obj = obj_rec;
	cv::Rect frm(0, 0, fWidth, fHeight);

	if ((obj&frm).area() < obj.area() / 3) return true;
	else return false;
}
void GMPHD_OGM::ClearOldEmptyTracklet(int current_fn, map<int, vector<BBTrk>>& tracklets, int MAXIMUM_OLD) {

	map<int, vector<BBTrk>> cleared_tracklets;

	vector<int> keys_old_vec;
	map<int, vector<BBTrk>>::iterator iter;

	for (iter = tracklets.begin(); iter != tracklets.end(); ++iter) {

		if (!iter->second.empty()) {
			if (iter->second.back().fn >= current_fn - MAXIMUM_OLD) {

				vector<BBTrk> track;

				vector<BBTrk>::iterator iterT;
				for (iterT = iter->second.begin(); iterT != iter->second.end(); ++iterT)
					track.push_back(iterT[0]);

				pair<map<int, vector<BBTrk>>::iterator, bool> isEmpty = cleared_tracklets.insert(map<int, vector<BBTrk>>::value_type(iter->first, track));

				if (isEmpty.second == false) { // already exists

				}
				else {

				}
			}
			else {
				keys_old_vec.push_back(iter->first);
			}
		}
		else {
			keys_old_vec.push_back(iter->first);
		}
	}

	// Swap and Clear Old Tracklets
	tracklets.clear();
	cleared_tracklets.swap(tracklets);
	for (iter = cleared_tracklets.begin(); iter != cleared_tracklets.end(); ++iter) {
		iter->second.clear();
	}
	cleared_tracklets.clear();

}
vector<vector<float>> GMPHD_OGM::ReturnTrackingResults(int iFrmCnt, vector<BBTrk>& liveReliabes) {
	
	vector<vector<float>> tracksResults;
	vector<BBTrk> trackBBs;

	if (iFrmCnt < this->iTotalFrames) {

		vector<BBTrk>::const_iterator iLive;
		for (iLive = liveReliabes.begin(); iLive != liveReliabes.end(); ++iLive) {
			if (iLive[0].isAlive) {
				vector<float> track;
				track.push_back((float)iLive[0].id);
				track.push_back((float)iLive[0].rec.x);
				track.push_back((float)iLive[0].rec.y);
				track.push_back((float)iLive[0].rec.width);
				track.push_back((float)iLive[0].rec.height);
				tracksResults.push_back(track);

				BBTrk trk;
				trk.id = iLive[0].id;
				trk.rec = iLive[0].rec;
				trk.fn = iFrmCnt;
				trackBBs.push_back(trk);
			}
		}

		this->allLiveReliables.push_back(trackBBs);
		trackBBs.clear();
	}
	if (GMPHD_TRACKER_MODE && this->params.FRAMES_DELAY_SIZE && (iFrmCnt+this->params.FRAMES_DELAY_SIZE+FRAME_OFFSET == this->iTotalFrames)) {
		for (int OFFSET = 1; OFFSET < this->params.TRACK_MIN_SIZE; OFFSET++) {
			vector<BBTrk>::iterator iterT;
			for (iterT = this->liveTracksBatch[OFFSET].begin(); iterT != this->liveTracksBatch[OFFSET].end(); ++iterT) {

				bool isReliable = false;
				vector<BBTrk>::iterator iterTR;
				for (iterTR = liveReliabes.begin(); iterTR != liveReliabes.end(); ++iterTR) {
					if (iterT->id == iterTR->id) {
						isReliable = true;
						// It Must be used
						iterTR->fn = iterT->fn;
						iterTR->id = iterT->id;
						iterTR->rec.x = iterT->rec.x;
						iterTR->rec.y = iterT->rec.y;
						iterTR->rec.width = iterT->rec.width;
						iterTR->rec.height = iterT->rec.height;
						break;
					}
				}
				// Select the targets which exists reliable targets vector.
				if (iterT->isAlive == true && isReliable) {

					cv::Rect tracking_output_rect = iterTR->rec;

					// Copy the tracking results for writing them in file
					vector<float> track;
					track.push_back((float)iterT->id);
					track.push_back((float)tracking_output_rect.x);
					track.push_back((float)tracking_output_rect.y);
					track.push_back((float)tracking_output_rect.width);
					track.push_back((float)tracking_output_rect.height);
					tracksResults.push_back(track);

					BBTrk trk;
					trk.id = iterT->id;
					trk.rec = tracking_output_rect;
					trk.fn = iterTR->fn; // (iFrmCnt + OFFSET);
					trackBBs.push_back(trk);
				}
			}
			this->allLiveReliables.push_back(trackBBs);
			trackBBs.clear();
		}
	}

	return tracksResults;
}