//
// Created by wei on 3/29/18.
//

#include "common/ConfigParser.h"
#include "common/data_transfer.h"
#include "core/SurfelWarpSerial.h"
#include "core/warp_solver/WarpSolver.h"
#include "core/geometry/SurfelNodeDeformer.h"
#include "visualization/Visualizer.h"
#include "imgproc/frameio/FetchInterface.h"
#include "imgproc/frameio/GenericFileFetch.h"

#include <thread>
#include <fstream>

// 构造函数
surfelwarp::SurfelWarpSerial::SurfelWarpSerial() {
	//The config is assumed to be updated
	const auto& config = ConfigParser::Instance();  // 单例模式
	
	//Construct the image processor
	FetchInterface::Ptr fetcher = std::make_shared<GenericFileFetch>(config.data_path());  // 有点意思，这个应该是用来读取图像的，获取所有图像的路径和数据
	m_image_processor = std::make_shared<ImageProcessor>(fetcher);  // 图像处理方法
	
	//Construct the holder for surfel geometry
	// 这里是用来处理surfel，包括融合，哪些点需要融合
	m_surfel_geometry[0] = std::make_shared<SurfelGeometry>();
	m_surfel_geometry[1] = std::make_shared<SurfelGeometry>();
	m_live_geometry_updater = std::make_shared<LiveGeometryUpdater>(m_surfel_geometry);
	
	//The warp field
	// 这里是用来解决旋转场的问题
	m_warp_field = std::make_shared<WarpField>();
	m_warpfield_initializer = std::make_shared<WarpFieldInitializer>();
	m_warpfield_extender = std::make_shared<WarpFieldExtender>();
	
	//The knn index
	// 用来做knn聚类，和新加入点的knn
	m_live_nodes_knn_skinner = KNNBruteForceLiveNodes::Instance();
	m_reference_knn_skinner = ReferenceNodeSkinner::Instance();
	
	//Construct the renderer
	m_renderer = std::make_shared<Renderer>(config.clip_image_rows(), config.clip_image_cols());
	
	//Map the resource into geometry
	m_renderer->MapSurfelGeometryToCuda(0, *(m_surfel_geometry[0]));  // 蜜汁操作，为什么映射到gpu之后又unmap，这里为什么要做一次这个操作呢
	m_renderer->MapSurfelGeometryToCuda(1, *(m_surfel_geometry[1]));
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);
	m_updated_geometry_index = -1; // None are updated
	
	//Constructs the solver
	m_warp_solver = std::make_shared<WarpSolver>();
	m_rigid_solver = std::make_shared<RigidSolver>();
	m_warp_solver->AllocateBuffer();

	//Construct the initializer and re-initializer
	m_geometry_initializer = std::make_shared<GeometryInitializer>();
	m_geometry_initializer->AllocateBuffer();
	m_geometry_reinit_processor = std::make_shared<GeometryReinitProcessor>(m_surfel_geometry);
	
	//The frame index
	m_frame_idx = config.start_frame_idx();
	m_reinit_frame_idx = m_frame_idx;
}

surfelwarp::SurfelWarpSerial::~SurfelWarpSerial() {
	//Release the explicit allocated buffer
	m_warp_solver->ReleaseBuffer();
	m_geometry_initializer->ReleaseBuffer();
}

/* The processing interface
 */
void surfelwarp::SurfelWarpSerial::ProcessFirstFrame() {
	//Process it
	// 获取了当前第一帧数据中的有效的surfel
	const auto surfel_array = m_image_processor->ProcessFirstFrameSerial(m_frame_idx);
	
	//Build the reference and live nodes, color and init time
	m_updated_geometry_index = 0;  // 这里应该是需要处理的
	// 渲染这里我先不管，因为这里感觉问题非常多，OpenGL虽然能够让我渲染，说实话也没啥问题，完全不影响我操作，反向传播与他二者是不会互相影响的
	// 渲染不能不管，这里问题很大呀，OpenGL虽然能够处理渲染的问题，但是前向和反向传播过程中的参数怎么办呢，这里也是问题呀
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	// 用有效的surfel_array来初始化m_surfel_geometry[m_updated_geometry_index]
	// m_surfel_geometry[m_updated_geometry_index]中的初始化是相同的
	m_geometry_initializer->InitFromObservationSerial(*m_surfel_geometry[m_updated_geometry_index], surfel_array);
	
	//Build the reference vertex and SE3 for the warp field
	const auto reference_vertex = m_surfel_geometry[m_updated_geometry_index]->GetReferenceVertexConfidence();  // 这里获取的每个点的顶点位置和置信度
	// 使用m_warpfield_initializer来初始化m_warp_field
	// m_warpfield_initializer是算法，m_warp_field是存储类
	m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(reference_vertex, m_warp_field);
	
	//Build the index and skinning nodes and surfels
	// 计算节点图中每个节点最近八个节点的距离和索引
	m_warp_field->BuildNodeGraph();
	
	//Perform skinning
	// reference_nodes是节点的坐标，所有的节点的坐标，w为1
	const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
	// 这里就是把节点导入到m_reference_knn_skinner的device里
	m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);
	// 返回当前帧参考帧的每个点的坐标和置信度数组和每个数据点的knn和knn weight
	auto skinner_geometry = m_surfel_geometry[m_updated_geometry_index]->SkinnerAccess();
	// 返回当前的node节点和节点的knn和knn weight
	auto skinner_warpfield = m_warp_field->SkinnerAccess();
	// 这里就要进行绑定了呀
	m_reference_knn_skinner->PerformSkinning(skinner_geometry, skinner_warpfield);

	//Unmap it
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	
	//Update the index
	m_frame_idx++;
}

void surfelwarp::SurfelWarpSerial::ProcessNextFrameWithReinit(bool offline_save) {
	//Draw the required maps, assume the buffer is not mapped to cuda at input
	const auto num_vertex = m_surfel_geometry[m_updated_geometry_index]->NumValidSurfels();
	const float current_time = m_frame_idx - 1;
	const Matrix4f init_world2camera = m_camera.GetWorld2CameraEigen();

	//Check the frame and draw
	SURFELWARP_CHECK(m_frame_idx >= m_reinit_frame_idx);
	const bool draw_recent = shouldDrawRecentObservation();
	if(draw_recent) {
		m_renderer->DrawSolverMapsWithRecentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	}
	else {
		m_renderer->DrawSolverMapsConfidentObservation(num_vertex, m_updated_geometry_index, current_time, init_world2camera);
	}
	
	//Map to solver maps
	Renderer::SolverMaps solver_maps;
	m_renderer->MapSolverMapsToCuda(solver_maps);
	m_renderer->MapSurfelGeometryToCuda(m_updated_geometry_index);
	
	//Process the next depth frame
	CameraObservation observation;
	//m_image_processor->ProcessFrameSerial(observation, m_frame_idx);
	m_image_processor->ProcessFrameStreamed(observation, m_frame_idx);
	
	//First perform rigid solver
	m_rigid_solver->SetInputMaps(solver_maps, observation, m_camera.GetWorld2Camera());
	const mat34 solved_world2camera = m_rigid_solver->Solve();
	m_camera.SetWorld2Camera(solved_world2camera);
	
	//The resource from geometry attributes
	const auto solver_geometry = m_surfel_geometry[m_updated_geometry_index]->SolverAccess();
	const auto solver_warpfield = m_warp_field->SolverAccess();
	
	//Pass the input to warp solver
	m_warp_solver->SetSolverInputs(
		observation,
		solver_maps,
		solver_geometry,
		solver_warpfield,
		m_camera.GetWorld2Camera() //The world to camera might be updated by rigid solver
	);
	
	//Solve it
	//m_warp_solver->SolveSerial();
	m_warp_solver->SolveStreamed();
	const auto solved_se3 = m_warp_solver->SolvedNodeSE3();
	
	//Do a forward warp and build index
	m_warp_field->UpdateHostDeviceNodeSE3NoSync(solved_se3);
	SurfelNodeDeformer::ForwardWarpSurfelsAndNodes(*m_warp_field, *m_surfel_geometry[m_updated_geometry_index], solved_se3);

	//Compute the nodewise error
	m_warp_solver->ComputeAlignmentErrorOnNodes();
	
	//Build the live node index for later used
	const auto live_nodes = m_warp_field->LiveNodeCoordinates();
	m_live_nodes_knn_skinner->BuildIndex(live_nodes);
	
	//Draw the map for point fusion
	m_renderer->UnmapSurfelGeometryFromCuda(m_updated_geometry_index);
	m_renderer->UnmapSolverMapsFromCuda();
	m_renderer->DrawFusionMaps(num_vertex, m_updated_geometry_index, m_camera.GetWorld2CameraEigen());
	
	//Map the fusion map to cuda
	Renderer::FusionMaps fusion_maps;
	m_renderer->MapFusionMapsToCuda(fusion_maps);
	//Map both maps to surfelwarp as they are both required
	m_renderer->MapSurfelGeometryToCuda(0);
	m_renderer->MapSurfelGeometryToCuda(1);
	
	//The hand tune variable now. Should be replaced later
	const bool use_reinit = shouldDoReinit();
	const bool do_integrate = shouldDoIntegration();

	//The geometry index that both fusion and reinit will write to, if no writing then keep current geometry index
	auto fused_geometry_idx = m_updated_geometry_index;
	
	//Depends on should do reinit or integrate
	if(use_reinit) {
		//First setup the idx
		m_reinit_frame_idx = m_frame_idx;
		fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

		//Hand in the input to reinit processor
		m_geometry_reinit_processor->SetInputs(
			fusion_maps,
			observation,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);
		
		//Process it
		const auto node_error = m_warp_solver->GetNodeAlignmentError();
		unsigned num_remaining_surfel, num_appended_surfel;
		m_geometry_reinit_processor->ProcessReinitObservedOnlySerial(num_remaining_surfel, num_appended_surfel);
		//m_geometry_reinit_processor->ProcessReinitNodeErrorSerial(num_remaining_surfel, num_appended_surfel, node_error, 0.06f);
		
		//Reinit the warp field
		const auto reference_vertex = m_surfel_geometry[fused_geometry_idx]->GetReferenceVertexConfidence();
		m_warpfield_initializer->InitializeReferenceNodeAndSE3FromVertex(reference_vertex, m_warp_field);
		
		//Build the index and skinning nodes and surfels
		m_warp_field->BuildNodeGraph();
		
		//Build skinning index
		const auto& reference_nodes = m_warp_field->ReferenceNodeCoordinates();
		m_reference_knn_skinner->BuildInitialSkinningIndex(reference_nodes);
		
		//Perform skinning
		auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
		auto skinner_warpfield = m_warp_field->SkinnerAccess();
		m_reference_knn_skinner->PerformSkinning(skinner_geometry, skinner_warpfield);
	} else if(do_integrate) {
		//Update the frame idx
		fused_geometry_idx = (m_updated_geometry_index + 1) % 2;

		//Hand in the input to fuser
		const auto warpfield_input = m_warp_field->GeometryUpdaterAccess();
		m_live_geometry_updater->SetInputs(
			fusion_maps,
			observation,
			warpfield_input,
			m_live_nodes_knn_skinner,
			m_updated_geometry_index,
			float(m_frame_idx),
			m_camera.GetWorld2Camera()
		);
		
		//Do fusion
		unsigned num_remaining_surfel, num_appended_surfel;
		//m_live_geometry_updater->ProcessFusionSerial(num_remaining_surfel, num_appended_surfel);
		m_live_geometry_updater->ProcessFusionStreamed(num_remaining_surfel, num_appended_surfel);
		
		//Do a inverse warping
		SurfelNodeDeformer::InverseWarpSurfels(*m_warp_field, *m_surfel_geometry[fused_geometry_idx], solved_se3);
		
		//Extend the warp field reference nodes and SE3
		const auto prev_node_size = m_warp_field->CheckAndGetNodeSize();
		const float4* appended_vertex_ptr = m_surfel_geometry[fused_geometry_idx]->ReferenceVertexArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<float4> appended_vertex_view(appended_vertex_ptr, num_appended_surfel);
		const ushort4* appended_knn_ptr = m_surfel_geometry[fused_geometry_idx]->SurfelKNNArray().RawPtr() + num_remaining_surfel;
		DeviceArrayView<ushort4> appended_surfel_knn(appended_knn_ptr, num_appended_surfel);
		m_warpfield_extender->ExtendReferenceNodesAndSE3Sync(appended_vertex_view, appended_surfel_knn, m_warp_field);
		
		//Rebuild the node graph
		m_warp_field->BuildNodeGraph();
		
		//Update skinning
		if(m_warp_field->CheckAndGetNodeSize() > prev_node_size){
			m_reference_knn_skinner->UpdateBruteForceSkinningIndexWithNewNodes(m_warp_field->ReferenceNodeCoordinates().DeviceArrayReadOnly(), prev_node_size);
			
			//Update skinning
			auto skinner_geometry = m_surfel_geometry[fused_geometry_idx]->SkinnerAccess();
			auto skinner_warpfield = m_warp_field->SkinnerAccess();
			m_reference_knn_skinner->PerformSkinningUpdate(skinner_geometry, skinner_warpfield, prev_node_size);
		}
	}
	
	//Unmap attributes
	m_renderer->UnmapFusionMapsFromCuda();
	m_renderer->UnmapSurfelGeometryFromCuda(0);
	m_renderer->UnmapSurfelGeometryFromCuda(1);
	
	//Debug save
	if(offline_save) {
		const auto with_recent = draw_recent || use_reinit;
		const auto& save_dir = createOrGetDataDirectory(m_frame_idx);
		saveCameraObservations(observation, save_dir);
		saveSolverMaps(solver_maps, save_dir);

		const auto num_fused_vertex = m_surfel_geometry[fused_geometry_idx]->NumValidSurfels();
		saveVisualizationMaps(
			num_fused_vertex, fused_geometry_idx,
			m_camera.GetWorld2CameraEigen(), m_camera.GetInitWorld2CameraEigen(),
			save_dir, with_recent
		);
	}
	
	//Update the index
	m_frame_idx++;
	m_updated_geometry_index = fused_geometry_idx;
}



/* The method for offline visualization
 */
void surfelwarp::SurfelWarpSerial::saveCameraObservations(
	const surfelwarp::CameraObservation &observation,
	const boost::filesystem::path &save_dir
) {
	//Save the segment mask
	Visualizer::SaveSegmentMask(observation.foreground_mask, observation.normalized_rgba_map, (save_dir / "foreground_mask.png").string(), 1);

	//Save the clipped, filtered depth image
	Visualizer::SaveDepthImage(observation.filter_depth_img, (save_dir / "cliped_filter_depth.png").string());

	//Save the raw depth image
	Visualizer::SaveDepthImage(observation.raw_depth_img, (save_dir / "raw_depth.png").string());
}


void surfelwarp::SurfelWarpSerial::saveCorrespondedCloud(
	const CameraObservation &observation,
	unsigned vao_idx,
	const boost::filesystem::path &save_dir
) {
	const std::string cloud_1_name = (save_dir / "observation.off").string();
	const std::string cloud_2_name = (save_dir / "model.off").string();
	
	auto geometry = m_surfel_geometry[vao_idx]->Geometry();
	Visualizer::SaveMatchedCloudPair(
		observation.vertex_config_map,
		geometry.live_vertex_confid.ArrayView(),
		m_camera.GetCamera2WorldEigen(),
		cloud_1_name, cloud_2_name
	);
	
	//Also save the reference point cloud
	Visualizer::SavePointCloud(geometry.reference_vertex_confid.ArrayView(), (save_dir / "reference.off").string());
}


void surfelwarp::SurfelWarpSerial::saveSolverMaps(
	const surfelwarp::Renderer::SolverMaps &solver_maps,
	const boost::filesystem::path &save_dir
) {
	//Save the rendered albedo maps
	Visualizer::SaveNormalizeRGBImage(solver_maps.normalized_rgb_map, (save_dir / "rendered_rgb.png").string());

	//Save the index map
	Visualizer::SaveValidIndexMap(solver_maps.index_map, 1, (save_dir / "validity_index_map.png").string());
	
	//Save the computed alignment error map
	m_warp_solver->ComputeAlignmentErrorMapDirect();
	cudaTextureObject_t alignment_error_map = m_warp_solver->GetAlignmentErrorMap();
	Visualizer::SaveGrayScaleImage(alignment_error_map, (save_dir / "alignment_error_map_direct.png").string(), 10.0f);

	//Compute again using nodes
	m_warp_solver->ComputeAlignmentErrorMapFromNode();
	alignment_error_map = m_warp_solver->GetAlignmentErrorMap();
	Visualizer::SaveGrayScaleImage(alignment_error_map, (save_dir / "alignment_error_map_node.png").string(), 10.0f);
	
	//Do statistic about the error value
	const auto node_error = m_warp_solver->GetNodeAlignmentError();
	std::ofstream error_output;
	error_output.open((save_dir/"node_error.txt").string());
	node_error.errorStatistics(error_output);
	error_output.close();
}


void surfelwarp::SurfelWarpSerial::saveVisualizationMaps(
	unsigned int num_vertex,
	int vao_idx,
	const Eigen::Matrix4f &world2camera,
	const Eigen::Matrix4f& init_world2camera,
	const boost::filesystem::path &save_dir,
	bool with_recent
) {
	m_renderer->SaveLiveNormalMap(num_vertex, vao_idx, m_frame_idx, world2camera, (save_dir / "live_normal.png").string(), with_recent);
	m_renderer->SaveLiveAlbedoMap(num_vertex, vao_idx, m_frame_idx, world2camera, (save_dir / "live_albedo.png").string(), with_recent);
	m_renderer->SaveLivePhongMap (num_vertex, vao_idx, m_frame_idx, world2camera, (save_dir /  "live_phong.png").string(), with_recent);

	m_renderer->SaveReferenceNormalMap(num_vertex, vao_idx, m_frame_idx, init_world2camera, (save_dir / "reference_normal.png").string(), with_recent);
	m_renderer->SaveReferenceAlbedoMap(num_vertex, vao_idx, m_frame_idx, init_world2camera, (save_dir / "reference_albedo.png").string(), with_recent);
	m_renderer->SaveReferencePhongMap (num_vertex, vao_idx, m_frame_idx, init_world2camera, (save_dir /  "reference_phong.png").string(), with_recent);
}

boost::filesystem::path surfelwarp::SurfelWarpSerial::createOrGetDataDirectory(int frame_idx) {
	//Construct the path
	boost::filesystem::path result_folder("frame_" + std::to_string(frame_idx));
	if(!boost::filesystem::exists(result_folder)) {
		boost::filesystem::create_directory(result_folder);
	}
	
	//The directory should be always exist
	return result_folder;
}


// The decision function for integrate and reinit
// Currently not implemented
bool surfelwarp::SurfelWarpSerial::shouldDoIntegration() const {
	return true;
}

static bool shouldDoReinitConfig(int frame_idx) {
	using namespace surfelwarp;
	const auto& config = ConfigParser::Instance();

	// Check the config
	if (!config.use_periodic_reinit()) {
		return false;
	}
	
	// Check the peroid
	const auto period = config.reinit_period();
	SURFELWARP_CHECK(period > 0);
	if (frame_idx > 0 && (frame_idx % period) == 0) {
		return true;
	}
	else
		return false;
}

//The processing interface of pipeline
bool surfelwarp::SurfelWarpSerial::shouldDoReinit() const {
	return shouldDoReinitConfig(m_frame_idx);
}


bool surfelwarp::SurfelWarpSerial::shouldDrawRecentObservation() const {
	return m_frame_idx - m_reinit_frame_idx <= Constants::kStableSurfelConfidenceThreshold + 1;
}