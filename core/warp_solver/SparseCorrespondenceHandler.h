//
// Created by wei on 4/5/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/surfel_types.h"
#include "common/algorithm_types.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"
#include <memory>

namespace surfelwarp {
	
	class SparseCorrespondenceHandler {
	private:
		//The info from depth observation
		struct {
			DeviceArrayView<ushort4> correspond_pixel_pairs;  // 观测到的gpc模型对应计算的像素对
			cudaTextureObject_t depth_vertex_map;  // 观测到的顶点图
		} m_observations;
		
		//The information from renderer
		struct {
			cudaTextureObject_t reference_vertex_map;  // reference的顶点图
			cudaTextureObject_t index_map;  // reference的对应的顶点索引，这个像素位置存的是哪个像素
			DeviceArrayView2D<KNNAndWeight> knn_map;  // knn的图
		} m_geometry_maps;

		//The input from warp field
		DeviceArrayView<DualQuaternion> m_node_se3;  // 节点图，保存的是每个节点的SE3，双四元数表示旋转平移
		mat34 m_camera2world;  // c
		
	public:
		using Ptr = std::shared_ptr<SparseCorrespondenceHandler>;
		SURFELWARP_DEFAULT_CONSTRUCT_DESTRUCT(SparseCorrespondenceHandler);
		SURFELWARP_NO_COPY_ASSIGN_MOVE(SparseCorrespondenceHandler);
		
		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The processing interface
		void SetInputs(
			DeviceArrayView<DualQuaternion> node_se3,
			DeviceArrayView2D<KNNAndWeight> knn_map,
			cudaTextureObject_t depth_vertex_map,
			DeviceArrayView<ushort4> correspond_pixel_pairs,
			//The rendered maps
			cudaTextureObject_t reference_vertex_map,
			cudaTextureObject_t index_map,
			const mat34& world2camera
		);
		
		//Update the node se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);

		
		/* Some pixels from the correspondence finder might
		 * not has a reference vertex on that pixel
		 */
	private:
		DeviceBufferArray<unsigned> m_valid_pixel_indicator;  // 用于显示哪个像素是有效的
		PrefixSum m_valid_pixel_prefixsum;  // 执行前缀和，剔除无用的像素对
		DeviceBufferArray<ushort4> m_corrected_pixel_pairs;  // 用于保存对应的像素对
	public:
		void ChooseValidPixelPairs(cudaStream_t stream = 0);
	
		
		/* Collect the valid depth/reference vertex and their knn
		 */
	private:
		DeviceBufferArray<float4> m_valid_target_vertex;  // 存储的是世界坐标系下的坐标，也就是当前输入帧的坐标通过相机外参做了一次转换
		DeviceBufferArray<float4> m_valid_reference_vertex;  // 就是reference的坐标
		DeviceBufferArray<ushort4> m_valid_vertex_knn;  // 对应的knn
		DeviceBufferArray<float4> m_valid_knn_weight;  // 对应的knn weight

		//The page-locked memory
		unsigned* m_correspondence_array_size;
	public:
		void CompactQueryPixelPairs(cudaStream_t stream = 0);
		void QueryCompactedArraySize(cudaStream_t stream = 0);
		void BuildCorrespondVertexKNN(cudaStream_t stream = 0);


		/* Perform a forward warp on the vertex for efficient computation
		 */
	private:
		DeviceBufferArray<float4> m_valid_warped_vertex;
		void forwardWarpFeatureVertex(cudaStream_t stream = 0);
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);
	
	public:
		Point2PointICPTerm2Jacobian Term2JacobianMap() const;
		DeviceArrayView<ushort4> SparseFeatureKNN() const { return m_valid_vertex_knn.ArrayView(); }
	};
	
}