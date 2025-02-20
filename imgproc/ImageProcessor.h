#pragma once

#include <memory>
#include <texture_types.h>
#include "common/macro_utils.h"
#include "common/common_types.h"
#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "common/surfel_types.h"
#include "common/algorithm_types.h"
#include "common/DeviceBufferArray.h"
#include "common/CameraObservation.h"
#include "math/device_mat.h"
#include "imgproc/frameio/FetchInterface.h"
#include "imgproc/segmentation/ForegroundSegmenter.h"
#include "imgproc/correspondence/ImagePairCorrespondence.h"


namespace surfelwarp {
	
	/**
	 * \brief The image processor fetch depth and color image,
	 *        produce struct ProcessedFrame as output. The threaded
	 *        mechanism is not implemented in this class (but outside),
	 *        as a ring buffer producer-consumer system.
	 *        This class does NOT maintain a frame index
	 */
	class ImageProcessor {
	public:
		using Ptr = std::shared_ptr<ImageProcessor>;

		//Constructor allocate the buffer, while de-constructor release them
		explicit ImageProcessor(const FetchInterface::Ptr& fetcher);
		~ImageProcessor();
		SURFELWARP_NO_COPY_ASSIGN_MOVE(ImageProcessor);

		//This is not thread-safe
		DeviceArrayView<DepthSurfel> ProcessFirstFrameSerial(size_t frame_idx, cudaStream_t stream = 0);
		void ProcessFrameSerial(CameraObservation& observation, size_t frame_idx, cudaStream_t stream = 0);

		/**
		* \brief The global data member
		*/
    private:
        //Size and intrinsic of the camera
        unsigned m_raw_img_cols, m_raw_img_rows;  // 原始宽高
		unsigned m_clip_img_cols, m_clip_img_rows;  // 裁剪后的宽高
		unsigned m_clip_near, m_clip_far;  // 裁剪后的前后距离，这个距离应该可以作为近平面和远平面

		//The intrinsic parameters
		Intrinsic m_raw_depth_intrinsic;  // 深度传感器内参
		Intrinsic m_raw_rgb_intrinsic;  // 相机内参
		Intrinsic m_clip_rgb_intrinsic;  // 裁剪后的图像对应的内参
		mat34 m_depth2rgb;  // 这是一个旋转平移矩阵3*4矩阵，这个矩阵目前是单位矩阵，作用推测可能有两种：1.相机和深度传感器没有对齐；2.外参
	public:
		unsigned clip_rows() const { return m_clip_img_rows; }
		unsigned clip_cols() const { return m_clip_img_cols; }


		/**
		* \brief Fetch the image from the fetcher.
		*        Use page-locked memory for async-loading.
		*/
    private:
        //The image fetcher from the constructor
        FetchInterface::Ptr m_image_fetcher;  // 用于获取图像，这是一个类

        //The opencv matrix and the underline memory for image fetching
        cv::Mat m_depth_img, m_rgb_img, m_rgb_img_prev;  // opencv矩阵，分别放置深度图，rgb图像和前一帧的rgb图像，用于数据加载
        void* m_depth_buffer_pagelock;  // 在cuda的固定区域，用于放置深度图像
        void* m_rgb_buffer_pagelock;  // 在cuda的固定区域，用于放置rgb图像
		void* m_rgb_prev_buffer_pagelock;  // 在cuda的固定区域，用于放置前一帧的rgb图像

        //Init the buffer for image fetch
        void allocateFetchBuffer();
        void releaseFetchBuffer();
    public:
        void FetchFrame(size_t frame_idx);
		void FetchDepthImage(size_t frame_idx);
		void FetchRGBImage(size_t frame_idx);
		void FetchRGBPrevFrame(size_t curr_frame_idx);
        const cv::Mat& RawDepthImageCPU() const { return m_depth_img; }
        const cv::Mat& RawRGBImageCPU() const { return m_rgb_img; }



		/**
		* \brief Upload the depth image to raw texture.
		*        Reproject the depth image into RGB frame,
		*        and perform cliping and filtering on it.
		*/
	private:
		//The texture for raw image and the underlying array
		//In the original size (not the cliped size)
		CudaTextureSurface m_depth_raw_collect;  // 虽然没太看明白，但是应该还是存储的深度数据，只是通过cuda方式进行处理

		//The Reproject depth image to color frame
		//Using a scale factor to avoid collision
		DeviceArray2D<unsigned short> m_reprojected_buffer;  // 用于存储投影后的深度数据，放大了两倍，防止发生碰撞

		//The texture, surface and array for filter depth image
		//In the cliped size
		CudaTextureSurface m_depth_filter_collect; // 虽然没太看明白，但是应该还是存储的深度数据，裁剪过后的，只是通过cuda方式进行处理

		//Init the buffer for depth texture
		void allocateDepthTexture();
		void releaseDepthTexture();
	public:
		void UploadDepthImage(cudaStream_t stream = 0);
		void ReprojectDepthToRGB(cudaStream_t stream = 0);
		void ClipFilterDepthImage(cudaStream_t stream = 0);
		cudaTextureObject_t RawDepthTexture() const { return m_depth_raw_collect.texture; }
		cudaTextureObject_t FilteredDepthTexture() const { return m_depth_filter_collect.texture; }



		/**
		* \brief Upload the rgb image to gpu memory.
		*        Perform clip, normalization and create density map.
		*        From this point, the intrinsic should be 
		*        m_clip_intrinsic_rgb
		*/
	private:
		//The flatten buffer for rgb image
		DeviceArray<uchar3> m_raw_rgb_buffer;  // 创建一个数组，存储rgb图像
		DeviceArray<uchar3> m_raw_rbg_buffer_prev;  // 创建一个数组，存储前一帧的rgb图像

		//The clipped and normalized rgb image, float4
		CudaTextureSurface m_clip_normalize_rgb_collect;  // 创建一个数组，存储裁剪后的图像 cuda相关
		CudaTextureSurface m_clip_normalize_rgb_collect_prev;  // 创建一个数组，存储前一帧的裁剪后的图像 cuda相关

		//The density map for rgb and rgb_prev, float1
		CudaTextureSurface m_density_map_collect;  // 密度图他大爷，这就是灰度图像，将rgb转换为了灰度值这个是密度相关的图，通过cuda来搞的，据说有加速效果
		CudaTextureSurface m_filter_density_map_collect;  // 对灰度图滤波一下，什么鬼，这个是密度相关的图，通过cuda来搞的，据说有加速效果
		
		//Init and destroy the buffer
		void allocateRGBBuffer();
		void releaseRGBBuffer();
	public:
		void UploadRawRGBImage(cudaStream_t stream = 0);
		void ClipNormalizeRGBImage(cudaStream_t stream = 0);
		const DeviceArray<uchar3>& RawRGBImageGPU() const { return m_raw_rgb_buffer; }
		cudaTextureObject_t ClipNormalizedRGBTexture() const { return m_clip_normalize_rgb_collect.texture; }
		cudaTextureObject_t ClipNormalizedRGBTexturePrev() const { return m_clip_normalize_rgb_collect_prev.texture; }
		cudaTextureObject_t DensityMapTexture() const { return m_density_map_collect.texture; }

		/**
		* \brief Compute geometry maps from depth image
		*        Use the CLIPED image and intrinsic
		*/
	private:
		//float4 texture, (x, y, z) is vertex, w is the confidence value
		CudaTextureSurface m_vertex_confid_collect;  // 顶点、置信度

		//float4 texture, (x, y, z) is normal, w is the radius
		CudaTextureSurface m_normal_radius_collect;  // 法线、半径

		//Create the destroy texture collect above
		void allocateGeometryTexture();
		void releaseGeometryTexture();
	public:
		void BuildVertexConfigMap(cudaStream_t stream = 0);
		void BuildNormalRadiusMap(cudaStream_t stream = 0);
		cudaTextureObject_t VertexConfidTexture() const { return m_vertex_confid_collect.texture; }
		cudaTextureObject_t NormalRadiusTexture() const { return m_normal_radius_collect.texture; }


        /**
		* \brief Compute the color_time maps from rgb image.
        *        Shoule be the same format as the surfel array
		*/
    private:
	    // float4，第一位是颜色，第二位是0暂时还不知道是什么，第三位是初始时间，第四位是也用初始时间赋值
        CudaTextureSurface m_color_time_collect;  // color time指的是颜色和时间，时间就是当前是多少帧，颜色就是点的颜色
        void allocateColorTimeTexture();
        void releaseColorTimeTexture();
    public:
        void BuildColorTimeTexture(size_t frame_idx, cudaStream_t stream = 0);
        cudaTextureObject_t ColorTimeTexture() const { return m_color_time_collect.texture; }


		/**
		* \brief Collect the valid depth surfel into array
		*/
	private:
		//Pre-allocate memory and array
		DeviceBufferArray<DepthSurfel> m_depth_surfel;  // 这是一个数组，但是这个数组存储的是surfel的信息，包含相关的surfel属性

		//The selector and collected valid depth surfel
		FlagSelection m_valid_depth_pixel_selector;  // 这个是选择器，用于选择出有效的深度点，具体怎么选择还要看代码实现
		
		void allocateValidSurfelSelectionBuffer();
		void releaseValidSurfelSelectionBuffer();
	public:
		void CollectValidDepthSurfel(cudaStream_t stream = 0);
		const DeviceArray<DepthSurfel> ValidDepthSurfelArray() { return m_depth_surfel.Array(); }
		
		
		/**
		* \brief The foreground segmentation mask and its filtered. In segmentation
		*        mask, the foreground is one; while in the filtered mask, the
		*        foreground is zero as it is used in optimization pipeline.
		*/
	private:
		//Need to explicit allocate/deallocate
		ForegroundSegmenter::Ptr m_foreground_segmenter;  // 分割器，用于分割出前景目标
		
		//Allocator and de-allocator
		void allocateForegroundSegmentationBuffer();
		void releaseForegroundSegmentationBuffer();
	public:
		//This method doesn't block the host thread that invoke this method.
		//For both naive implementation and permutdral implementation.
		void SegmentForeground(int frame_idx, cudaStream_t stream = 0);
		cudaTextureObject_t ForegroundMask() const { return m_foreground_segmenter->ForegroundMask(); }
		cudaTextureObject_t FilterForegroundMask() const { return m_foreground_segmenter->FilterForegroundMask(); }
		
		
		/**
		* \brief Sparse feature algorithm. Need foreground mask as input
		*/
	private:
		ImagePairCorrespondence::Ptr m_feature_correspondence_finder;  // 其实就是一种光流方法，获取点的对应关系，GPC
		void allocateFeatureCorrespondenceBuffer();
		void releaseFeatureCorrespondenceBuffer();
	public:
		//This method might block the host thread that invoke the method.
		void FindCorrespondence(cudaStream_t stream = 0);
		DeviceArray<ushort4> CorrespondencePixelPair() const {
			return m_feature_correspondence_finder->CorrespondedPixelPairs();
		}

		/**
		* \brief Compute the image gradient of the density image and the filtered
		*        foreground mask. The gradient map are all float2 maps, x and y components
		*        is the gradient w.r.t x and y axis.
		*/
	// 忘记这个密度图和前景mask图的分别代表什么意思了，密度图是不是就是滤波后的图像，前景mask是不是就是加上mask后的图像
	private:
	    // 我去，我靠，为啥给foreground_mask搞一个gradient map呀
		CudaTextureSurface m_foreground_mask_gradient_map_collect;  // 计算一个梯度，正好可以用在头发丝重建
		CudaTextureSurface m_density_gradient_map_collect;  // 计算一个梯度，正好能用在头发丝重建
		void allocateGradientMap();
		void releaseGradientMap();
	public:
		void ComputeGradientMap(cudaStream_t stream = 0);
		cudaTextureObject_t DensityGradientTexture() const { return m_density_gradient_map_collect.texture; }
		cudaTextureObject_t ForegroundMaskGradientTexture() const { return m_foreground_mask_gradient_map_collect.texture; }

		/**
		 * \beief Do the image processing in streamed mode. Totally 3 stream is required.
		 *        The stream is allocated inside the processor during construction.
		 */
	// 通过三个流来处理数据，这些流是可以并行处理，有个好处就是可以并行处理不同的任务，加速
	// 或者是处理前后帧？？？
	private:
		cudaStream_t m_processor_stream[3];
		void initProcessorStream();
		void releaseProcessorStream();
		void syncAllProcessorStream();
	public:
		void ProcessFrameStreamed(CameraObservation& observation, size_t frame_idx);
	};
}
