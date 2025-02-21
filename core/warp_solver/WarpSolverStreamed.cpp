#include "core/warp_solver/WarpSolver.h"

void surfelwarp::WarpSolver::initSolverStream() {
	//Create the stream
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[0]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[1]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[2]));
	cudaSafeCall(cudaStreamCreate(&m_solver_stream[3]));
	
	//Hand in the stream to pcg solver
	UpdatePCGSolverStream(m_solver_stream[0]);
}

void surfelwarp::WarpSolver::releaseSolverStream() {
	//Update 0 stream to pcg solver
	UpdatePCGSolverStream(0);

	cudaSafeCall(cudaStreamDestroy(m_solver_stream[0]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[1]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[2]));
	cudaSafeCall(cudaStreamDestroy(m_solver_stream[3]));

	//Assign to null stream
	m_solver_stream[0] = 0;
	m_solver_stream[1] = 0;
	m_solver_stream[2] = 0;
}

void surfelwarp::WarpSolver::syncAllSolverStream() {
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));
}

void surfelwarp::WarpSolver::SolveStreamed() {
	//Sync before compuation
	syncAllSolverStream();

	//Actual computation
	buildSolverIndexStreamed();
	// 最多只会求解五次
	for(auto i = 0; i < Constants::kNumGaussNewtonIterations; i++) {
		// 默认将密度设置为了0，所以这里应该就是
		if (m_iteration_data.IsGlobalIteration())
			solverIterationGlobalIterationStreamed();
		else
			solverIterationLocalIterationStreamed();
	}

	//Sync again for debug
	syncAllSolverStream();
}

void surfelwarp::WarpSolver::buildSolverIndexStreamed() {
	// 这里比较有意思了，这里把每个顶点给渲染到了屏幕上，然后根据这个屏幕上的像素，来确定每个位置所用到的knn坐标和knn权重，牛逼
	QueryPixelKNN(m_solver_stream[0]); //Sync is required here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));

	//FetchPotentialDenseImageTermPixelsFixedIndex
	// 这里只保存有用的节点，无用的节点就暂时不处理了
	m_image_knn_fetcher->SetInputs(m_knn_map, m_rendered_maps.index_map);
	m_image_knn_fetcher->MarkPotentialMatchedPixels(m_solver_stream[0]);
	m_image_knn_fetcher->CompactPotentialValidPixels(m_solver_stream[0]);
	//m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize();

	//FindPotentialForegroundMaskPixelSynced();
	// 获取有效的像素
	setDensityForegroundHandlerFullInput();
	m_density_foreground_handler->MarkValidColorForegroundMaskPixels(m_solver_stream[1]);
	m_density_foreground_handler->CompactValidMaskPixel(m_solver_stream[1]);
	//m_density_foreground_handler->QueryCompactedMaskPixelArraySize();

	//SelectValidSparseFeatureMatchedPairs();
	// 计算像素对
	SetSparseFeatureHandlerFullInput();
	m_sparse_correspondence_handler->ChooseValidPixelPairs(m_solver_stream[2]);
	m_sparse_correspondence_handler->CompactQueryPixelPairs(m_solver_stream[2]);
	//m_sparse_correspondence_handler->QueryCompactedArraySize();

	//The sync group
	m_image_knn_fetcher->SyncQueryCompactedPotentialPixelSize(m_solver_stream[0]); //Sync is inside the method
	m_density_foreground_handler->QueryCompactedMaskPixelArraySize(m_solver_stream[1]); //Sync is inside the method
	m_sparse_correspondence_handler->QueryCompactedArraySize(m_solver_stream[2]); //Sync is inside the method
	setDenseDepthHandlerFullInput();
	setDensityForegroundHandlerFullInput();

	//Before the index part: A sync happened here
	SetNode2TermIndexInput();
	BuildNode2TermIndex(m_solver_stream[0]); //This doesnt block
	BuildNodePair2TermIndexBlocked(m_solver_stream[1]); //This will block
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}


void surfelwarp::WarpSolver::solverIterationGlobalIterationStreamed() {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianFixedIndex(m_solver_stream[0], m_solver_stream[1], m_solver_stream[2], m_solver_stream[3]); //A sync should happened here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
	BuildPreconditionerGlobalIteration(m_solver_stream[0]);
	ComputeJtResidualGlobalIteration(m_solver_stream[1]);
	MaterializeJtJNondiagonalBlocksGlobalIteration(m_solver_stream[2]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));

	//The assemble of matrix: a sync here
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	//Debug methods
	//LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}

void surfelwarp::WarpSolver::solverIterationLocalIterationStreamed() {
	//Hand in the new SE3 to handlers
	m_dense_depth_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_density_foreground_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());
	m_sparse_correspondence_handler->UpdateNodeSE3(m_iteration_data.CurrentWarpFieldInput());

	//The computation of jacobian
	ComputeTermJacobianFixedIndex(m_solver_stream[0], m_solver_stream[1], m_solver_stream[2], m_solver_stream[3]); // A sync should happend here
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[3]));

	//The computation of diagonal blks JtJ and JtError
	SetPreconditionerBuilderAndJtJApplierInput();
	SetJtJMaterializerInput();
	BuildPreconditioner(m_solver_stream[0]);
	ComputeJtResidual(m_solver_stream[1]);
	MaterializeJtJNondiagonalBlocks(m_solver_stream[2]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[1]));
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[2]));
	

	//The assemble of matrix: a sync here
	const auto diagonal_blks = m_preconditioner_rhs_builder->JtJDiagonalBlocks();
	m_jtj_materializer->AssembleBinBlockCSR(diagonal_blks, m_solver_stream[0]);

	//Debug methods
	//LOG(INFO) << "The total squared residual in materialized, fixed-index solver is " << ComputeTotalResidualSynced(m_solver_stream[0]);

	//Solve it and update
	SolvePCGMaterialized();
	m_iteration_data.ApplyWarpFieldUpdate(m_solver_stream[0]);
	cudaSafeCall(cudaStreamSynchronize(m_solver_stream[0]));
}