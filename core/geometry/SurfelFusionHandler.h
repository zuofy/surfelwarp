//
// Created by wei on 5/2/18.
//

#pragma once

#include "common/CameraObservation.h"
#include "common/DeviceBufferArray.h"
#include "common/algorithm_types.h"
#include "math/device_mat.h"
#include "core/SurfelGeometry.h"
#include "core/render/Renderer.h"

namespace surfelwarp {
	
	// The task of surfel fusion handler is: given the fusion
	// geometry and the fusion maps, fuse it to existing geometry
	// and compute indicators for which surfels should remain and
	// which depth pixel will potentiall be appended to the surfel array
	// This kernel parallelize over images (not the surfel array)
	class SurfelFusionHandler {
	private:
		//Basic parameters
		unsigned m_image_rows, m_image_cols;
		
		//The input from outside
		Renderer::FusionMaps m_fusion_maps;  // 应该是要融合的点的信息
		SurfelGeometry::SurfelFusionInput m_fusion_geometry;  // 要融合的几何信息，包含更加全面的数据，甚至包含knn的信息
		CameraObservation m_observation;
		float m_current_time;
		mat34 m_world2camera;
		bool m_use_atomic_append; //Use atomic append or not
		
		//The buffer maintained by this class
		DeviceBufferArray<unsigned> m_remaining_surfel_indicator;  // 用来保存surfel的指示器，具体干啥还不知道，应该存储了一些点，暂时保留吧
		
	public:
		using Ptr = std::shared_ptr<SurfelFusionHandler>;
		SurfelFusionHandler();
		~SurfelFusionHandler();
		SURFELWARP_NO_COPY_ASSIGN(SurfelFusionHandler);
		
		//The input requires all CamearObservation
		void SetInputs(
			const Renderer::FusionMaps& maps,
			const CameraObservation& observation,
			const SurfelGeometry::SurfelFusionInput& geometry,
			float current_time,
			const mat34& world2camera,
			bool use_atomic_append = true
		);
		
		//The processing interface for fusion
		void ProcessFusion(cudaStream_t stream = 0);
		void BuildCandidateAppendedPixelsSync(cudaStream_t stream = 0);

		//The fusion pipeline for reinit
		void ProcessFusionReinit(cudaStream_t stream = 0);
		
		//The fused indicator
		struct FusionIndicator {
			DeviceArraySlice<unsigned> remaining_surfel_indicator;
			DeviceArrayView<ushort2> appended_pixels;
		};
		FusionIndicator GetFusionIndicator();


		/* Process data fusion using compaction
		 */
	private:
		DeviceArray<unsigned> m_appended_depth_surfel_indicator; //This is fixed in size 要添加数据的指示器
		PrefixSum m_appended_surfel_indicator_prefixsum; // 用于处理前缀和，多看看，这里能不能参考3dgs修改
		DeviceBufferArray<ushort2> m_compacted_appended_pixel;  // 要添加的像素坐标？？
		void prepareFuserArguments(void* fuser_ptr);
		void processFusionAppendCompaction(cudaStream_t stream = 0);
		void processFusionReinit(cudaStream_t stream = 0);
		void compactAppendedIndicator(cudaStream_t stream = 0);
	public:
		void ZeroInitializeRemainingIndicator(unsigned num_surfels, cudaStream_t stream = 0);
		DeviceArraySlice<unsigned> GetRemainingSurfelIndicator();
		DeviceArrayView<unsigned> GetAppendedObservationCandidateIndicator() const;


		/* Process appedning using atomic operation
		 */
	private:
		unsigned* m_atomic_appended_pixel_index; // 原子操作？？
		DeviceBufferArray<ushort2> m_atomic_appended_observation_pixel;
		void processFusionAppendAtomic(cudaStream_t stream = 0);
		void queryAtomicAppendedPixelSize(cudaStream_t stream = 0);


		//The debug method for fusion pipeline
	private:
		void fusionStatistic(bool using_atomic = false);
		void confidenceStatistic();
	};
	
	
}

