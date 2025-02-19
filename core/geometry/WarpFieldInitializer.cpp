//
// Created by wei on 5/10/18.
//

#include "common/Constants.h"
#include "core/geometry/WarpFieldInitializer.h"
#include "core/geometry/WarpFieldUpdater.h"
#include "core/geometry/VoxelSubsamplerSorting.h"

surfelwarp::WarpFieldInitializer::WarpFieldInitializer() {
	m_vertex_subsampler = std::make_shared<VoxelSubsamplerSorting>();
	m_vertex_subsampler->AllocateBuffer(Constants::kMaxNumSurfels);
	// 这里为什么会乘个Constants::kMaxSubsampleFrom，还是需要考虑下的
	m_node_candidate.AllocateBuffer(Constants::kMaxSubsampleFrom * Constants::kMaxNumNodes);
}


surfelwarp::WarpFieldInitializer::~WarpFieldInitializer() {
	m_vertex_subsampler->ReleaseBuffer();
}

void surfelwarp::WarpFieldInitializer::InitializeReferenceNodeAndSE3FromVertex(
	const DeviceArrayView<float4>& reference_vertex,
	WarpField::Ptr warp_field,
	cudaStream_t stream
) {
	// First subsampling
	// 看看怎么下采样的
	// 这里是分割为体素，找到一个体素中离体素中心最近的点，保存在m_node_candidate中
	performVertexSubsamplingSync(reference_vertex, stream);
	
	//Next select from candidate
	const auto& h_candidates = m_node_candidate.HostArray();
	// 再次挑选一些节点，如果两个节点的位置小于某个阈值，则只保留一个节点
	WarpFieldUpdater::InitializeReferenceNodesAndSE3FromCandidates(*warp_field, h_candidates, stream);
}

void surfelwarp::WarpFieldInitializer::performVertexSubsamplingSync(
	const DeviceArrayView<float4>& reference_vertex,
	cudaStream_t stream
) {
	// The voxel size
	// 采样的体素大小应该是比每个节点的半径要小的
	// 突然想知道flame的最小边长和最大边长是多少了
	const auto subsample_voxel = 0.7f * Constants::kNodeRadius;
	
	//Perform subsampling
	// 候选点吗，这里的大小是node的5倍
	auto& node_candidates = m_node_candidate;
	m_vertex_subsampler->PerformSubsample(reference_vertex, node_candidates, subsample_voxel, stream);
}