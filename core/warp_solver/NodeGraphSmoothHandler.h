//
// Created by wei on 5/17/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/ArrayView.h"
#include "common/DeviceBufferArray.h"
#include "math/DualQuaternion.hpp"
#include "core/warp_solver/solver_types.h"

#include <memory>

namespace surfelwarp {
	
	class NodeGraphSmoothHandler {
	private:
		//The input data from solver
		DeviceArrayView<DualQuaternion> m_node_se3;  // se3
		DeviceArrayView<ushort2> m_node_graph;  // 相邻八节点
		DeviceArrayView<float4> m_reference_node_coords;  // 节点的坐标

	public:
		using Ptr = std::shared_ptr<NodeGraphSmoothHandler>;
		NodeGraphSmoothHandler();
		~NodeGraphSmoothHandler();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(NodeGraphSmoothHandler);

		//The input interface from solver
		void SetInputs(
			const DeviceArrayView<DualQuaternion>& node_se3, 
			const DeviceArrayView<ushort2>& node_graph, 
			const DeviceArrayView<float4>& reference_nodes
		);

		//Do a forward warp on nodes
	private:
		DeviceBufferArray<float3> Ti_xj_;  // 公式10里的一项
		DeviceBufferArray<float3> Tj_xj_;  // 公式10的另一项
		DeviceBufferArray<unsigned char> m_pair_validity_indicator; // 标记该项目是否有用
		void forwardWarpSmootherNodes(cudaStream_t stream = 0);
	public:
		void BuildTerm2Jacobian(cudaStream_t stream = 0);
		NodeGraphSmoothTerm2Jacobian Term2JacobianMap() const;
	};


} // surfelwarp