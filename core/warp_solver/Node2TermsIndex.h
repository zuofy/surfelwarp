//
// Created by wei on 4/2/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "common/algorithm_types.h"
#include "core/warp_solver/term_offset_types.h"
#include <memory>

namespace surfelwarp {

	class Node2TermsIndex {
	private:
		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn; // 像素有效区域对应的knnEach depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;  // 节点图，每个节点最近的八个节点
			DeviceArrayView<ushort4> foreground_mask_knn; //The same as density term，前景mask有效区域最近的knn
			DeviceArrayView<ushort4> sparse_feature_knn; // 有效像素对光流对应的knnEach 4 nodes correspond to 3 scalar cost
		} m_term2node;

		//The number of nodes
		unsigned m_num_nodes;  // 有多少个节点
	
		//The term offset of term2node map
		TermTypeOffset m_term_offset;  // 把上边有效的像素的索引的坐标范围拼一下，m_term2node第一个像素有多少个值，第二个有多少个值，第三个有多少个值，第四个有多少个值
	public:
		//Accessed by pointer, default construct/destruct
		using Ptr = std::shared_ptr<Node2TermsIndex>;
		Node2TermsIndex();
		~Node2TermsIndex() = default;
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Node2TermsIndex);
	
		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The input
		void SetInputs(
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,  unsigned num_nodes,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn  = DeviceArrayView<ushort4>()
		);
		
		//The main interface
		void BuildIndex(cudaStream_t stream = 0);
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;


		/* Fill the key and value given the terms
		 */
	private:
		DeviceBufferArray<unsigned short> m_node_keys;  // knn对应的节点的索引
		DeviceBufferArray<unsigned> m_term_idx_values;  // m_node_keys在所有m_term2node项目里对应的索引
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);
		
		
		
		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned short, unsigned> m_node2term_sorter;  // 这里存储key value，按照m_node_keys拍了个序
		DeviceBufferArray<unsigned> m_node2term_offset;  // 这里边对应的node太多了，这里只求需要的节点
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);
		

		/* A series of checking functions
		 */
	private:
		static void check4NNTermIndex(int typed_term_idx, const std::vector<ushort4>& knn_vec, unsigned short node_idx);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned short node_idx);
		void compactedIndexSanityCheck();


		/* The accessing interface
		 * Depends on BuildIndex
		 */
	public:
		struct Node2TermMap {
			DeviceArrayView<unsigned> offset;
			DeviceArrayView<unsigned> term_index;
			TermTypeOffset term_offset;
		};
		
		//Return the outside-accessed index
		Node2TermMap GetNode2TermMap() const {
			Node2TermMap map;
			map.offset = m_node2term_offset.ArrayReadOnly();
			map.term_index = m_node2term_sorter.valid_sorted_value;
			map.term_offset = m_term_offset;
			return map;
		}
	};

}