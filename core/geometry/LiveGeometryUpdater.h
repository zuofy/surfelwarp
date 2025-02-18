//
// Created by wei on 5/2/18.
//

#pragma once

#include "core/WarpField.h"
#include "core/SurfelGeometry.h"
#include "core/geometry/SurfelFusionHandler.h"
#include "core/geometry/FusionRemainingSurfelMarker.h"
#include "core/geometry/AppendSurfelProcessor.h"
#include "core/geometry/DoubleBufferCompactor.h"
#include "imgproc/ImageProcessor.h"

namespace surfelwarp {
	
	/* The geometry updater access to both surfel geometry (but not friend),
	 * Its task is to update the LIVE geometry and knn for both existing
	 * and newly appended surfels. The double buffer approach is implemented here
	 */
	// 这个任务用来更新live的几何和knn对于已经存在的和新添加的surfel
	class LiveGeometryUpdater {
	private:
		SurfelGeometry::Ptr m_surfel_geometry[2];  // 这个其实就是m_surfel_geometry里的两个东西，指向的应该是同一块内存区域，现在还搞不明白为什么用两个
		//For any processing iteration, this variable should be constant, only assign by external variable
		// 在任何处理过程中，该值只能由外部复制，不应该被改变
		int m_updated_idx;  // 暂时还不知道是什么意思，但是应该是说是第几帧
		float m_current_time;  // 这个值应该是当前处理的是第几帧
		
		//The map from the renderer
		Renderer::FusionMaps m_fusion_maps;  // 变形后需要融合的点
		
		//The skinning method from updater
		WarpField::LiveGeometryUpdaterInput m_warpfield_input;
		KNNSearch::Ptr m_live_node_skinner;
		
		//The observation from depth camera
		CameraObservation m_observation;
		mat34 m_world2camera;
	public:
		using Ptr = std::shared_ptr<LiveGeometryUpdater>;
		explicit LiveGeometryUpdater(SurfelGeometry::Ptr surfel_geometry[2]);
		~LiveGeometryUpdater();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(LiveGeometryUpdater);
		
		//The process input
		void SetInputs(
			const Renderer::FusionMaps& maps,
			const CameraObservation& observation,
			const WarpField::LiveGeometryUpdaterInput& warpfield_input,
			const KNNSearch::Ptr& live_node_skinner,
			int updated_idx, float current_time,
			const mat34& world2camera
		);
		
		//The processing pipeline
		void TestFusion();
		void ProcessFusionSerial(unsigned& num_remaining_surfel, unsigned& num_appended_surfel, cudaStream_t stream = 0);
		
		/* The buffer and method for surfel fusion
		 */
	private:
		SurfelFusionHandler::Ptr m_surfel_fusion_handler;  // 用来融合live帧的点，这个是个狠人呀，但是暂时还不知具体怎么操作，后续再处理吧，这里先保持这样，里边很多概念暂时还看不明白
	public:
		void FuseCameraObservationSync(cudaStream_t stream = 0);
		
		
		/* The buffer and method for cleaning the existing surfels
		 */
	private:
		FusionRemainingSurfelMarker::Ptr m_fusion_remaining_surfel_marker;  // 这个是用来融合已经存在的数据，有些数据已经存在但是需要更新
	public:
		void MarkRemainingSurfels(cudaStream_t stream = 0);
		RemainingLiveSurfelKNN GetRemainingLiveSurfelKNN() const;
		
		
		/* The buffer and method to process appended surfels
		 */
	private:
		AppendSurfelProcessor::Ptr m_appended_surfel_processor;  // 没太理解，但是估计也是做数据处理的，用来对需要添加surfle
	public:
		void ProcessAppendedSurfels(cudaStream_t stream = 0);


		/* Compact the remaining surfel and appended surfel to another buffer
		 */
	private:
		DoubleBufferCompactor::Ptr m_surfel_compactor; // 这个按照我的理解是将处理的数据存储再另外一个buffer里
	public:
		void CompactSurfelToAnotherBufferSync(unsigned& num_remaining_surfel, unsigned& num_appended_surfel, cudaStream_t stream = 0);
		void TestCompactionKNNFirstIter(unsigned num_remaining_surfel, unsigned num_appended_surfel);
		
		
		/* The stream for fusion processing
		 */
	private:
		cudaStream_t m_fusion_stream[2];
		void initFusionStream();
		void releaseFusionStream();
	public:
		void ProcessFusionStreamed(unsigned& num_remaining_surfel, unsigned& num_appended_surfel);
	};
	
	
}