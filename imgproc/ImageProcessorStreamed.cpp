#include "imgproc/ImageProcessor.h"

void surfelwarp::ImageProcessor::initProcessorStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_processor_stream[2]));
}

void surfelwarp::ImageProcessor::releaseProcessorStream() {
	//Destroy these streams
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_processor_stream[2]));

	//Assign to null value
	m_processor_stream[0] = 0;
	m_processor_stream[1] = 0;
	m_processor_stream[2] = 0;
}

void surfelwarp::ImageProcessor::syncAllProcessorStream() {
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[2]));
}

void surfelwarp::ImageProcessor::ProcessFrameStreamed(CameraObservation & observation, size_t frame_idx) {
	FetchFrame(frame_idx);  // 加载图像
	UploadDepthImage(m_processor_stream[0]);  // 将深度图像转移到gpu上
	UploadRawRGBImage(m_processor_stream[0]);  // 将rgb图像转移到gpu上

	//This seems cause some problem ,disable it at first
	//ReprojectDepthToRGB(stream);

	ClipFilterDepthImage(m_processor_stream[0]);  // 滤波是干啥的，滤波就是将一个范围内的深度图用个高斯核卷积一下，感觉能够平滑一下点云
	ClipNormalizeRGBImage(m_processor_stream[0]);  // 归一化RGB图像，滤波灰度图像和前一帧的归一化RGB图像

	//The geometry map
	BuildVertexConfigMap(m_processor_stream[0]);  // 初始化了顶点和置信度的信息，但是置信度初始化为1，应该是信息还没处理完吧
	BuildNormalRadiusMap(m_processor_stream[0]);  // 初始化法线和半径
	BuildColorTimeTexture(frame_idx, m_processor_stream[0]);  // 初始化颜色和时间（颜色、0，时间，时间）

	//Sync here
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));

	//Invoke other expensive computations
	SegmentForeground(frame_idx, m_processor_stream[0]); //This doesn't block, even for hashing based method // 分割
	FindCorrespondence(m_processor_stream[1]); //This will block, thus sync inside  // GPC光流

	//The gradient map depends on filtered mask
	cudaSafeCall(cudaStreamSynchronize(m_processor_stream[0]));
	ComputeGradientMap(m_processor_stream[0]);

	//Sync and output
	syncAllProcessorStream();
	memset(&observation, 0, sizeof(observation));

	//The raw depth image for visualization
	observation.raw_depth_img = RawDepthTexture();

	//The geometry maps
	observation.filter_depth_img = FilteredDepthTexture();
	observation.vertex_config_map = VertexConfidTexture();
	observation.normal_radius_map = NormalRadiusTexture();

	//The color maps
	observation.color_time_map = ColorTimeTexture();
	observation.normalized_rgba_map = ClipNormalizedRGBTexture();
	observation.normalized_rgba_prevframe = ClipNormalizedRGBTexturePrev();
	observation.density_map = DensityMapTexture();
	observation.density_gradient_map = DensityGradientTexture();

	//The foreground masks
	observation.foreground_mask = ForegroundMask();
	observation.filter_foreground_mask = FilterForegroundMask();
	observation.foreground_mask_gradient_map = ForegroundMaskGradientTexture();

	//The correspondence pixel pairs
	const auto& pixel_pair_array = CorrespondencePixelPair();
	observation.correspondence_pixel_pairs = DeviceArrayView<ushort4>(pixel_pair_array.ptr(), pixel_pair_array.size());  // 还有对应的像素对
}