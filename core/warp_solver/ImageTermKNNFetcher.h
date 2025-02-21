//
// Created by wei on 7/4/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/algorithm_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/surfel_types.h"
#include "core/warp_solver/solver_types.h"

#include <memory>

namespace surfelwarp {
	
	
	class ImageTermKNNFetcher {
		//The info from config
		unsigned m_image_height;
		unsigned m_image_width;
		
		//The info from solver
		// 有点意思，这个东西的knn在图像空间的
		// 然后还跟着一个index_map，这个表示的是一个像素对应的顶点是哪个玩意儿
		struct {
			DeviceArrayView2D<KNNAndWeight> knn_map;
			cudaTextureObject_t index_map;
		} m_geometry_maps;
	
	public:
		//The contructor group
		using Ptr = std::shared_ptr<ImageTermKNNFetcher>;
		ImageTermKNNFetcher();
		~ImageTermKNNFetcher();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ImageTermKNNFetcher);
		
		//The input from the solver
		void SetInputs(
			const DeviceArrayView2D<KNNAndWeight>& knn_map,
			cudaTextureObject_t index_map
		);
		
		
		/* The main processor methods for mark all potential valid pixels
		 */
	private:
		//A fixed size array to indicator the pixel validity
		// 起始就是个标记，用于标记哪些像素是有用的，有用的像素为1，无用的像素为0
		DeviceArray<unsigned> m_potential_pixel_indicator;
	public:
		//This method, only collect pixel that has non-zero index map value
		//All these pixels are "potentially" matched with depth pixel with appropriate SE3
		// 标记哪些像素是有用的
		void MarkPotentialMatchedPixels(cudaStream_t stream = 0);


		//After all the potential pixels are marked
		// 这里的目的就是为了找到所有的有效像素，并且进行标记
	private:
		PrefixSum m_indicator_prefixsum;  // 有效像素前缀和，指明这是第几个有用的像素
		DeviceBufferArray<ushort2> m_potential_pixels;  // 一个数组，索引表示第几个有用像素，值表示像素的位置
		DeviceBufferArray<ushort4> m_dense_image_knn;  // 存储的是每个有用像素对应的节点的索引
		DeviceBufferArray<float4> m_dense_image_knn_weight;  // 存储的是每个有用像素对应的knn的权重
	public:
		void CompactPotentialValidPixels(cudaStream_t stream = 0);

	private:
		unsigned* m_num_potential_pixel;
	public:
		void SyncQueryCompactedPotentialPixelSize(cudaStream_t stream = 0);
		
		
		//Accessing interface
	public:
		struct ImageTermPixelAndKNN {
			DeviceArrayView<ushort2> pixels;
			DeviceArrayView<ushort4> node_knn;
			DeviceArrayView<float4> knn_weight;
		};
		ImageTermPixelAndKNN GetImageTermPixelAndKNN() const {
			ImageTermPixelAndKNN output;
			output.pixels = m_potential_pixels.ArrayReadOnly();
			output.node_knn = m_dense_image_knn.ArrayReadOnly();
			output.knn_weight = m_dense_image_knn_weight.ArrayReadOnly();
			return output;
		}
		DeviceArrayView<ushort4> DenseImageTermKNNArray() const { return m_dense_image_knn.ArrayView(); }
		
		
		//Sanity check
		void CheckDenseImageTermKNN(const DeviceArrayView<ushort4>& dense_depth_knn_gpu);
	};
	
	
} // surfelwarp