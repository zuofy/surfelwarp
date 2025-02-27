//
// Created by wei on 5/22/18.
//

#pragma once

#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/CameraObservation.h"
#include "common/SynchronizeArray.h"
#include "core/render/Renderer.h"

#include <memory>

namespace surfelwarp {
	
	
	class RigidSolver {
	private:
		//The intrinsic should be clipped intrinsic
		Intrinsic m_project_intrinsic;
		unsigned m_image_rows, m_image_cols;
		
		//The rendered map from renderer
		struct {
			cudaTextureObject_t live_vertex_map;  // live帧的顶点
			cudaTextureObject_t live_normal_map;  // live帧的法线
		} m_solver_maps;
		
		//The map from observation
		struct {
			cudaTextureObject_t vertex_map;  // 当前观察到的顶点图
			cudaTextureObject_t normal_map;  // 当前观察到的法线图
		} m_observation;
		
		//The initial transformation
		mat34 m_curr_world2camera;  // 上一帧的w2c方程
	
	public:
		using Ptr = std::shared_ptr<RigidSolver>;
		explicit RigidSolver();
		~RigidSolver();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(RigidSolver);
		
		//The input map from the solver
		void SetInputMaps(
			const Renderer::SolverMaps& solver_maps,
			const CameraObservation& observation,
			const mat34& init_world2camera
		);
		
		//The solver interface
		mat34 Solve(int max_iters = 3, cudaStream_t stream = 0);
		
		//The required macro
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	private:
		DeviceArray2D<float> m_reduce_buffer;
		SynchronizeArray<float> m_reduced_matrix_vector;  // 一个(21 + 6)的array是用来干什么的
		void allocateReduceBuffer();
		void rigidSolveDeviceIteration(cudaStream_t stream = 0);
		
		//The data required for host iteration
		Eigen::Matrix<float, 6, 6> JtJ_;
		Eigen::Matrix<float, 6, 1> JtErr_;
		void rigidSolveHostIterationSync(cudaStream_t stream = 0);
	};
	
}