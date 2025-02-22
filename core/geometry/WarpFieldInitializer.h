//
// Created by wei on 5/10/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/SynchronizeArray.h"
#include "core/geometry/VoxelSubsampler.h"
#include "core/WarpField.h"
#include <memory>

namespace surfelwarp {
	
	
	/**
	 * \brief The warp field initializer class takes input from reference vertex array,
	 * performing subsampling, and build the synched array of reference nodes and node SE3.
	 * The reference nodes requires more subsampling, and the SE3 is just identity.
	 * This operation is used by both the first frame and geometry reinitialization.
	 */
	// 仅用于第一帧和几何重新初始化
	// 目的是对参考帧创建warpfield
	// 方法是下采样点，然后给每个点初始一个单位矩阵作为SE3
	class WarpFieldInitializer {
	public:
		using Ptr = std::shared_ptr<WarpFieldInitializer>;
		WarpFieldInitializer();
		~WarpFieldInitializer();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(WarpFieldInitializer);

		//The processing interface
		void InitializeReferenceNodeAndSE3FromVertex(const DeviceArrayView<float4>& reference_vertex, WarpField::Ptr warp_field, cudaStream_t stream = 0);


		/* Perform subsampling of the reference vertex, fill the node to node_candidate
		 * Of course, this operation requires sync
		 */
	private:
		VoxelSubsampler::Ptr m_vertex_subsampler;  // 这里用了体素下采样，具体怎么操作还要再关注关注
		SynchronizeArray<float4> m_node_candidate;
		void performVertexSubsamplingSync(const DeviceArrayView<float4>& reference_vertex, cudaStream_t stream = 0);
	};
	
}
