//
// Created by wei on 3/18/18.
//

#pragma once

#include "core/render/glad/glad.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

//Cuda headers
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

//STL headers
#include <tuple>
#include <vector>
#include <Eigen/Eigen>
#include <memory>

//The type decals
#include "common/common_types.h"
#include "core/render/GLSurfelGeometryVBO.h"
#include "core/render/GLRenderedMaps.h"
#include "core/render/GLClearValues.h"
#include "core/render/GLShaderProgram.h"

namespace surfelwarp {
	
	
	class Renderer {
	private:
		//These member should be obtained from the config parser
		int m_image_width;  // 图像的宽，裁剪后的
		int m_image_height;  // 图像的高，裁剪后的
		int m_fusion_map_width;  // 这里宽是m_image_width的Constants::kFusionMapScale（4倍）
		int m_fusion_map_height;  // 这里高是m_image_height的Constants::kFusionMapScale（4倍）
		
		//The parameters that is accessed by drawing pipelines
		float4 m_renderer_intrinsic;  // 裁剪后的内参
		float4 m_width_height_maxdepth;  // 图像的宽，高，最大的深度
	public:
		//Accessed by pointer
		using Ptr = std::shared_ptr<Renderer>;
		explicit Renderer(int image_rows, int image_cols);
		~Renderer();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(Renderer);
		
		
		/* GLFW windows related variables and functions
		 */
	private:
	    // 这里就是物理意义上的显示器，可能又一个也可能有多个，会指定在哪个显示器进行显示
		GLFWmonitor* mGLFWmonitor = nullptr;
		// 指定显示窗口，窗口的名称，窗口的分辨率
		GLFWwindow* mGLFWwindow = nullptr;
		void initGLFW();
		
		
		/* The buffer and method to clear the image
		 */
	private:
	    // 这里设置的这些值都是用来清空缓冲区的
		// 我感觉这样可以直接通过这个来赋值
		GLClearValues m_clear_values;
		void initClearValues();
		
		
		/* The vertex buffer objects for surfel geometry
		 * Note that a double-buffer scheme is used here
		 */
	private:
	    // 这里的vertex buffer object就是surfel对象中的顶点位置，法线，颜色等信息
		// 这里采用了双缓冲区设置
		GLSurfelGeometryVBO m_surfel_geometry_vbos[2];
		void initVertexBufferObjects();
		void freeVertexBufferObjects();
	public:
		void MapSurfelGeometryToCuda(int idx, SurfelGeometry& geometry, cudaStream_t stream = 0);
		void MapSurfelGeometryToCuda(int idx, cudaStream_t stream = 0);
		void UnmapSurfelGeometryFromCuda(int idx, cudaStream_t stream = 0);
		

		/* The buffer for rendered maps
		 */
	private:
		//The frame/render buffer required for online processing
		GLFusionMapsFrameRenderBufferObjects m_fusion_map_buffers; // 一时之间有点蒙圈，感觉应该是用来融合reference和live的
		GLSolverMapsFrameRenderBufferObjects m_solver_map_buffers; // 一时之间有点蒙圈，感觉应该是用来求解reference和live的形变的
		
		//The frame/render buffer for offline visualization
		GLOfflineVisualizationFrameRenderBufferObjects m_visualization_draw_buffers; // 一时之间有点蒙圈，感觉应该是用来可视化的
		void initFrameRenderBuffers();
		void freeFrameRenderBuffers();
		
		
		/* The vao for rendering, must be init after
		 * the initialization of vbos
		 */
	private:
		//The vao for processing, correspond to double buffer scheme
		GLuint m_fusion_map_vao[2]; // 我推测是用来做融合的
		GLuint m_solver_map_vao[2]; // 我推测是用来做位姿解算的
		
		//The vao for offline visualization of reference and live geometry
		GLuint m_reference_geometry_vao[2]; // 只有参考帧的surfel
		GLuint m_live_geometry_vao[2];  // 包含参考帧和live帧的surfel
		void initMapRenderVAO();
		
		
		/* The shader program to render the maps for
		 * solver and geometry updater
		 */
	private:
		GLShaderProgram m_fusion_map_shader;  // 着色器语言怎么处理，感觉都是问题呀
		GLShaderProgram m_solver_map_shader;  // This shader will draw recent observation
		void initProcessingShaders();

		//the collect of shaders for visualization
		struct {
			GLShaderProgram normal_map;
			GLShaderProgram phong_map;
			GLShaderProgram albedo_map;
		} m_visualization_shaders;
		void initVisualizationShaders();
		void initShaders();
		
		//The workforce method for solver maps drawing
		void drawSolverMaps(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, bool with_recent_observation);

		//The workforce method for offline visualization
		void drawVisualizationMap(
			GLShaderProgram& shader, 
			GLuint geometry_vao, 
			unsigned num_vertex, int current_time, 
			const Matrix4f& world2camera, 
			bool with_recent_observation
		);

	public:
		void DrawFusionMaps(unsigned num_vertex, int vao_idx, const Matrix4f& world2camera);
		void DrawSolverMapsConfidentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera);
		void DrawSolverMapsWithRecentObservation(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera);

		//The offline visualization methods
		void SaveLiveNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveLiveAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveLivePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferenceNormalMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferenceAlbedoMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		void SaveReferencePhongMap(unsigned num_vertex, int vao_idx, int current_time, const Matrix4f& world2camera, const std::string& path, bool with_recent = true);
		
		//Debug drawing functions
		void DebugFusionMapsDraw(unsigned num_vertex, int vao_idx);
		void DebugSolverMapsDraw(unsigned num_vertex, int vao_idx);
		
		
		
		/* The access of fusion map
		 */
	public:
		struct FusionMaps {
			cudaTextureObject_t warp_vertex_map;
			cudaTextureObject_t warp_normal_map;
			cudaTextureObject_t index_map;
			cudaTextureObject_t color_time_map;
		};
		void MapFusionMapsToCuda(FusionMaps& maps, cudaStream_t stream = 0);
		void UnmapFusionMapsFromCuda(cudaStream_t stream = 0);
		
		/* The access of solver maps
		 */
	public:
		struct SolverMaps {
			cudaTextureObject_t reference_vertex_map;  // 就是参考帧的顶点
			cudaTextureObject_t reference_normal_map;  // 就是参考帧的法线
			cudaTextureObject_t warp_vertex_map;  // 就是活动帧的顶点
			cudaTextureObject_t warp_normal_map;  // 就是活动帧的法线2
			cudaTextureObject_t index_map;  // 就是每个对应的顶点序号
			cudaTextureObject_t normalized_rgb_map;  // 就是归一化之后的rgb值
		};
		void MapSolverMapsToCuda(SolverMaps& maps, cudaStream_t stream = 0);
		void UnmapSolverMapsFromCuda(cudaStream_t stream = 0);
	};
	
} // namespace surfelwarp