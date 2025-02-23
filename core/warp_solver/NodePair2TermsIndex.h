//
// Created by wei on 4/16/18.
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
	
	class NodePair2TermsIndex {
	public:
		using Ptr = std::shared_ptr<NodePair2TermsIndex>;
		NodePair2TermsIndex();
		~NodePair2TermsIndex() = default;
		SURFELWARP_NO_COPY_ASSIGN(NodePair2TermsIndex);
		
		//Explicit allocate/de-allocate
		void AllocateBuffer();
		void ReleaseBuffer();
		
		//The input for index
		void SetInputs(
			unsigned num_nodes,
			DeviceArrayView<ushort4> dense_image_knn,
			DeviceArrayView<ushort2> node_graph,
			//These costs might be empty
			DeviceArrayView<ushort4> foreground_mask_knn = DeviceArrayView<ushort4>(),
			DeviceArrayView<ushort4> sparse_feature_knn  = DeviceArrayView<ushort4>()
		);
		
		//The operation interface
		void BuildHalfIndex(cudaStream_t stream = 0);
		void QueryValidNodePairSize(cudaStream_t stream = 0); //Will block the stream
		unsigned NumTerms() const;
		unsigned NumKeyValuePairs() const;
		
		//Build the symmetric and row index
		void BuildSymmetricAndRowBlocksIndex(cudaStream_t stream = 0);
		
		//The access interface
		struct NodePair2TermMap {
			DeviceArrayView<unsigned> encoded_nodepair;
			DeviceArrayView<uint2> nodepair_term_range;
			DeviceArrayView<unsigned> nodepair_term_index;
			TermTypeOffset term_offset;
			//For bin-block csr
			DeviceArrayView<unsigned> blkrow_offset;
			DeviceArrayView<int> binblock_csr_rowptr;
			const int* binblock_csr_colptr;
		};
		NodePair2TermMap GetNodePair2TermMap() const;
		
		
		/* Fill the key and value given the terms
		 */
	private:
		//The input map from terms to nodes, the input might be empty for dense_density, foreground mask and sparse feature
		struct {
			DeviceArrayView<ushort4> dense_image_knn; // 有效像素对应的位置 Each depth scalar term has 4 nearest neighbour
			DeviceArrayView<ushort2> node_graph;  // 每个节点最近的八个节点
			//DeviceArrayView<ushort4> density_map_knn; //Each density scalar term has 4 nearest neighbour
			DeviceArrayView<ushort4> foreground_mask_knn; //The same as density term 前景mask区域有效的区域
			DeviceArrayView<ushort4> sparse_feature_knn; //Each 4 nodes correspond to 3 scalar cost 光流匹配有效的区域
		} m_term2node;
		
		//The term offset of term2node map
		TermTypeOffset m_term_offset;  // 每个m_term2node对应的size大小，存储的是这四个数组的大小
		unsigned m_num_nodes;  // 节点的数量
		
		/* The key-value buffer for indexing
		 */
	private:
		DeviceBufferArray<unsigned> m_nodepair_keys;  // 这里还是有点不一样的，这里保存的是成对的匹配，比如knn有四个节点，这里就从四个节点挑出来两个，小在前大在后，也就是6 1选中
		DeviceBufferArray<unsigned> m_term_idx_values;  // 对应m_term2node中从上到小的排列
	public:
		void buildTermKeyValue(cudaStream_t stream = 0);
		
		
		/* Perform key-value sort, do compaction
		 */
	private:
		KeyValueSort<unsigned, unsigned> m_nodepair2term_sorter;  // 按照m_nodepair_keys对所有的匹配对进行排序
		DeviceBufferArray<unsigned> m_segment_label;  // 标记哪个是有效的，就是剔除重复的
		PrefixSum m_segment_label_prefixsum;  // 标记的前缀和，挺有趣的，感觉这个前缀和是个万金油呀
		
		//The compacted half key and values
		DeviceBufferArray<unsigned> m_half_nodepair_keys;  // 存储有效的nodepair2term
		DeviceBufferArray<unsigned> m_half_nodepair2term_offset;  // 这个有效的nodepair2term在m_nodepair2term_sorter的索引
	public:
		void sortCompactTermIndex(cudaStream_t stream = 0);
	
		
		/* Fill the other part of the matrix
		 */
	private:
		DeviceBufferArray<unsigned> m_compacted_nodepair_keys;  // 这里为什么要把节点ij重新用节点ji存储一遍，是为了方便计算吗
		DeviceBufferArray<uint2> m_nodepair_term_range;  // 把m_half_nodepair2term_offset逐一元素复制两次
		KeyValueSort<unsigned, uint2> m_symmetric_kv_sorter;  // 把这个m_compacted_nodepair_keys拍了个序
	public:
		void buildSymmetricCompactedIndex(cudaStream_t stream = 0);
		
		
		/* Compute the offset and length of each BLOCKED row
		 */
	private:
	    // 0-10是第一个节点对应的元素，11-15是第二个节点对应的元素，数组的大小为节点的数量加1
		// m_blkrow_offset_array其实就是存的是所有的节点，每个节点对应的范围
		DeviceBufferArray<unsigned> m_blkrow_offset_array;
		// 数组大小和节点数量是一样的
		// 每个位置存每个节点对应了多少个节点对
		DeviceBufferArray<unsigned> m_blkrow_length_array;
		void blockRowOffsetSanityCheck();
		void blockRowLengthSanityCheck();
	public:
		void computeBlockRowLength(cudaStream_t stream = 0);
		
		
		/* Compute the map from block row to the elements in this row block
		 */
	private:
		// 这里的idx是(m_num_nodes * 6) / 32，最大设置为1024，每六个节点分为一组，求每组哪个对应有最多的元素
		// 将最多的元素的数量乘6保存到m_binlength_array中，这里是在分组吗
		DeviceBufferArray<unsigned> m_binlength_array;  // 大小为(m_num_nodes * 6) / 32，是用于CSR列索引
		// 我只能瞎扯了，暂时就理解为上边分组之后每组多少个元素，每组对应的非0元素数量？？？？？
		// 这是在做压缩吗，感觉得把代码运行起来看看了
		DeviceBufferArray<unsigned> m_binnonzeros_prefixsum; // 大小为(m_num_nodes * 6) / 32 + 1，用于CSR的行索引
		DeviceBufferArray<int> m_binblocked_csr_rowptr;   // 说实话我真看不懂了，这里是干啥的，给我绕迷糊了
		void binLengthNonzerosSanityCheck();
		void binBlockCSRRowPtrSanityCheck();
	public:
		void computeBinLength(cudaStream_t stream = 0);
		void computeBinBlockCSRRowPtr(cudaStream_t stream = 0);
		
		
		/* Compute the column ptr for bin block csr matrix
		 */
	private:
		DeviceBufferArray<int> m_binblocked_csr_colptr;  // 去他妈的，就假如是把这些东西分了个块吧
		void binBlockCSRColumnPtrSanityCheck();
	public:
		void nullifyBinBlockCSRColumePtr(cudaStream_t stream = 0);
		void computeBinBlockCSRColumnPtr(cudaStream_t stream = 0);
		
		
		
		/* Perform sanity check for nodepair2term
		 */
	public:
		void CheckHalfIndex();
		void CompactedIndexSanityCheck();
		
		//Check the size and distribution of the size of index
		void IndexStatistics();
		
		//Check whether the smooth term contains nearly all index
		//that can be exploited to implement more efficient indexing
		//Required download data and should not be used in real-time code
		void CheckSmoothTermIndexCompleteness();
	private:
		static void check4NNTermIndex(int typed_term_idx,
		                       const std::vector<ushort4> &knn_vec,
		                       unsigned encoded_nodepair);
		static void checkSmoothTermIndex(int smooth_term_idx, const std::vector<ushort2>& node_graph, unsigned encoded_nodepair);
	};
	
}
