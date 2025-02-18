//
// Created by wei on 3/29/18.
//

#pragma once

#include "common/Constants.h"
#include "common/ConfigParser.h"
#include "imgproc/ImageProcessor.h"
#include "imgproc/frameio/VolumeDeformFileFetch.h"
#include "core/SurfelGeometry.h"
#include "core/WarpField.h"
#include "core/Camera.h"
#include "core/render/Renderer.h"
#include "core/geometry/LiveGeometryUpdater.h"
#include "core/geometry/GeometryReinitProcessor.h"
#include "core/geometry/GeometryInitializer.h"
#include "core/geometry/KNNBruteForceLiveNodes.h"
#include "core/geometry/ReferenceNodeSkinner.h"
#include "core/geometry/WarpFieldInitializer.h"
#include "core/geometry/WarpFieldExtender.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/warp_solver/RigidSolver.h"

#include <boost/filesystem.hpp>

namespace surfelwarp {
	// 这个类就牛批了，这个类是总的处理的类，所有的代码都是从这里开始处理的
	// 整个算法的核心，来吧，啃硬骨头了
	class SurfelWarpSerial {
	private:
		//The primary components
		ImageProcessor::Ptr m_image_processor;  // 数据处理的一些东西
		Renderer::Ptr m_renderer;  // 这个是glfw实现的，用于可视乎渲染数据
		
		//The surfel geometry and their updater
		SurfelGeometry::Ptr m_surfel_geometry[2];  // 创建了两个surfle，应该是用于cuda编程的时候缓冲区交换的，或者有其他作用，待考虑
		int m_updated_geometry_index;
		LiveGeometryUpdater::Ptr m_live_geometry_updater;
		
		//The warp field and its updater
		WarpField::Ptr m_warp_field;  // 用于存储形变场
		WarpFieldInitializer::Ptr m_warpfield_initializer; // 用于进行形变场的初始化
		WarpFieldExtender::Ptr m_warpfield_extender;  // 用于新增节点对形变场进行优化
		
		//The camera(SE3 transform)
		Camera m_camera;  // 相机的外参
		
		//The knn index for live and reference nodes
		KNNBruteForceLiveNodes::Ptr m_live_nodes_knn_skinner;  // knn绑定
		ReferenceNodeSkinner::Ptr m_reference_knn_skinner;  // 用于绑定更新和reinitialization更新knn
		
		//The warp solver
		WarpSolver::Ptr m_warp_solver;   // 用于求解非刚性形变的
		RigidSolver::Ptr m_rigid_solver;  // 用于求解刚性形变的
		//::WarpSolver::Ptr m_legacy_solver;
		
		//The component for geometry processing
		GeometryInitializer::Ptr m_geometry_initializer;
		GeometryReinitProcessor::Ptr m_geometry_reinit_processor;
		
		//The frame counter
		int m_frame_idx;  // 当前要处理的帧序号
		int m_reinit_frame_idx;  // 需要重新初始化的帧序号
		
	public:
		using Ptr = std::shared_ptr<SurfelWarpSerial>;
		SurfelWarpSerial();  // 先啃最硬的骨头，构造函数
		~SurfelWarpSerial();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(SurfelWarpSerial);
		
		//Process the first frame
		void ProcessFirstFrame();
		void ProcessNextFrameNoReinit();
		void ProcessNextFrameWithReinit(bool offline_save = true);
		//void ProcessNextFrameLegacySolver();
		
		//The testing methods
		void TestGeometryProcessing();
		void TestSolver();
		void TestSolverWithRigidTransform();
		void TestRigidSolver();
		void TestPerformance();

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
		
		
		/* The method to save the informaiton for offline visualization/debug
		 * Assume the geometry pipeline can be called directly.
		 * These methods should be disabled on Real-Time code
		 */
	private:
		//The camera observation
		void saveCameraObservations(const CameraObservation& observation, const boost::filesystem::path& save_dir);
		
		//The rendered solver maps, required the same cuda context (but not OpenGL context)
		void saveSolverMaps(const Renderer::SolverMaps& solver_maps, const boost::filesystem::path& save_dir);
		
		//Save the coorresponded geometry and obsertion
		void saveCorrespondedCloud(const CameraObservation& observation, unsigned vao_idx, const boost::filesystem::path& save_dir);
		
		
		//The rendered and shaded geometry, This method requires access to OpenGL pipeline
		void saveVisualizationMaps(
			unsigned num_vertex,
			int vao_idx,
			const Eigen::Matrix4f& world2camera,
			const Eigen::Matrix4f& init_world2camera,
			const boost::filesystem::path& save_dir,
			bool with_recent = true
		);
	
		//The directory for this iteration
		static boost::filesystem::path createOrGetDataDirectory(int frame_idx);

		//The decision function for integration and reinit
		bool shouldDoIntegration() const;
		bool shouldDoReinit() const;
		bool shouldDrawRecentObservation() const;
	};
	
}