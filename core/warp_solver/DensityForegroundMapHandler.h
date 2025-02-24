//
// Created by wei on 3/31/18.
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
#include "core/warp_solver/ImageTermKNNFetcher.h"
#include <memory>

namespace surfelwarp {
	
	class DensityForegroundMapHandler {
	private:
		//The info from config
		int m_image_height;
		int m_image_width;
		Intrinsic m_project_intrinsic;  // 内参
		
		//The info from solver
		DeviceArrayView<DualQuaternion> m_node_se3; // 当前reference的节点对应的双四元数
		DeviceArrayView2D<KNNAndWeight> m_knn_map; // 每个像素点对应的knn节点坐标和权重
		mat34 m_world2camera;  // 外参
		
		//The maps from depth observation
		struct {
			cudaTextureObject_t foreground_mask;  // 输入帧得前景mask
			cudaTextureObject_t filtered_foreground_mask;  // 输入帧滤波后的前景mask
			cudaTextureObject_t foreground_mask_gradient_map;  // 输入帧滤波后的前景mask梯度图

			//The density map from depth
			cudaTextureObject_t density_map;  // 输入帧对应的灰度图
			cudaTextureObject_t density_gradient_map;  // 输入帧对应的灰度图对应的梯度图
		} m_depth_observation;
		

		//The map from renderer
		struct {
			cudaTextureObject_t reference_vertex_map;  // reference的顶点
			cudaTextureObject_t reference_normal_map;  // reference的法线
			cudaTextureObject_t index_map;  // reference的索引
			cudaTextureObject_t normalized_rgb_map;  // reference的归一化RGB
		} m_geometry_maps;
		
		//The pixel from the indexer
		//包含三个部分
		// {1.pixels 一个数组，索引表示第几个有用像素，值表示像素的位置
		//  2.node_knn 存储的是每个有用像素对应的节点的索引
		//  3.存储的是每个有用像素对应的knn的权重}
		ImageTermKNNFetcher::ImageTermPixelAndKNN m_potential_pixels_knn; 
	
	public:
		using Ptr = std::shared_ptr<DensityForegroundMapHandler>;
		DensityForegroundMapHandler();
		~DensityForegroundMapHandler() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(DensityForegroundMapHandler);

		//Explicit allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		

		//Set input
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3,  // 当前的节点对应的双四元数
			const DeviceArrayView2D<KNNAndWeight>& knn_map,  // 每个像素点对应的knn节点坐标和权重
			//The foreground mask terms
			cudaTextureObject_t foreground_mask, // 前景mask
			cudaTextureObject_t filtered_foreground_mask,  // 滤波后的前景mask
			cudaTextureObject_t foreground_gradient_map,  // 滤波后的前景mask梯度图
			//The color density terms
			cudaTextureObject_t density_map,  // 灰度图
			cudaTextureObject_t density_gradient_map,  // 灰度图的梯度图
			//The rendered maps
			cudaTextureObject_t reference_vertex_map,  // 参考的顶点
			cudaTextureObject_t reference_normal_map,  // 参考的法线
			cudaTextureObject_t index_map,  // 每个像素对应的顶点
			cudaTextureObject_t normalized_rgb_map,  // 每个像素对应的归一化坐标
			const mat34& world2camera,  // 当前的相机w2c矩阵
			//The potential pixels,
			const ImageTermKNNFetcher::ImageTermPixelAndKNN& potential_pixels_knn  // 妈的，我崩溃了，这个就是上一步求得稠密的knn那些东西
		);
		
		//Update the node se3
		void UpdateNodeSE3(DeviceArrayView<DualQuaternion> node_se3);
		

		//The finder interface
		void FindValidColorForegroundMaskPixels(cudaStream_t color_stream = 0, cudaStream_t mask_stream = 0);
		void FindPotentialForegroundMaskPixelSynced(cudaStream_t stream = 0);
		

		/* Mark the valid pixel for both color and foreground mask
		 */
	private:
		//These should be 2D maps
		DeviceArray<unsigned> m_color_pixel_indicator_map;  // 某个位置的周围都有点的索引，那这个位置就是有效的
		DeviceArray<unsigned> m_mask_pixel_indicator_map;  // 某个位置的前景mask不为0，这个位置也是有效的
	public:
		void MarkValidColorForegroundMaskPixels(cudaStream_t stream = 0);
		
		
		/* The compaction for maps
		 */
	private:
		PrefixSum m_color_pixel_indicator_prefixsum;
		PrefixSum m_mask_pixel_indicator_prefixsum;  // 前缀和，为了筛选出来有用的点
		DeviceBufferArray<ushort2> m_valid_color_pixel, m_valid_mask_pixel;  // 一个数组，索引表示第几个有用像素，值表示像素的位置，有效的像素和mask
		DeviceBufferArray<ushort4> m_valid_color_pixel_knn, m_valid_mask_pixel_knn;  // 存储的是每个有用像素对应的节点的索引，有效的像素和mask
		DeviceBufferArray<float4> m_valid_color_pixel_knn_weight, m_valid_mask_pixel_knn_weight;  // 存储的是每个有用像素对应的knn的权重，有效的像素和mask

		//The pagelocked memory
		unsigned* m_num_mask_pixel;  // 有多少个mask内有用的像素
	public:
		void CompactValidColorPixel(cudaStream_t stream = 0);
		void QueryCompactedColorPixelArraySize(cudaStream_t stream = 0);
		void CompactValidMaskPixel(cudaStream_t stream = 0);
		void QueryCompactedMaskPixelArraySize(cudaStream_t stream = 0);
		
		
		/* Compute the gradient
		 */
	private:
		DeviceBufferArray<float> m_color_residual;  // 密度差异
		DeviceBufferArray<TwistGradientOfScalarCost> m_color_twist_gradient;  // 颜色梯度？密度梯度？
		DeviceBufferArray<float> m_foreground_residual; // 和密度计算是一样的，这里将mask作为密度
		DeviceBufferArray<TwistGradientOfScalarCost> m_foreground_twist_gradient;  // 和密度计算是一样的，这里将mask作为密度
		void computeDensityMapTwistGradient(cudaStream_t stream = 0);
		void computeForegroundMaskTwistGradient(cudaStream_t stream = 0);
	public:
		void ComputeTwistGradient(cudaStream_t color_stream, cudaStream_t foreground_stream);
		void Term2JacobianMaps(
			DensityMapTerm2Jacobian& density_term2jacobian,
			ForegroundMaskTerm2Jacobian& foreground_term2jacobian
		);

		
		/* The access interface
		 */
	public:
		DeviceArrayView<ushort4> DensityTermKNN() const { return m_valid_color_pixel_knn.ArrayView(); }
		DeviceArrayView<ushort4> ForegroundMaskTermKNN() const { return m_valid_mask_pixel_knn.ArrayView(); }
	};
	
}