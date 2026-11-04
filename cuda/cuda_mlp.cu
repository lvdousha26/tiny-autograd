// ---------------------------------------------------------------------------
// cuda_mlp.cu —— 用 CUDA 重写 MNIST 上的 784 -> 256 -> 10 单隐层 MLP 训练。
//
// 和 C++ 引擎（Tensor/Net）完全独立：手写 GEMM、ReLU、softmax 交叉熵、SGD 的
// kernel，不依赖 cuBLAS / cuDNN，只用到 cudart。目的是回答一个问题：
// 同一套「前向 - 反向 - 更新」的数学，搬到 GPU 上到底快多少，代价是什么。
//
// 数据、超参、损失函数与 main.cpp 的 MNIST_DATA_mlp 保持一致：
//   输入按 255 归一化，交叉熵 + softmax，梯度对 batch 取均值，朴素 SGD。
//
// 编译与运行见 cuda/README.md（本机靠 conda 装的 nvcc 12.6 + VS2022 的 cl.exe）。
//   ./cuda_mlp.exe --epochs 10 --batch 256 --lr 0.3
//   ./cuda_mlp.exe --bench-gemm        # 单独测 GEMM 吞吐
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

// 所有 kernel 走同一条自建 stream：既方便统一异步，也是录 CUDA Graph 的前提
// （legacy 默认流不能用来做 capture）。
static cudaStream_t g_stream = nullptr;

#define CK(x)                                                                        \
	do                                                                               \
	{                                                                                \
		cudaError_t err_ = (x);                                                      \
		if (err_ != cudaSuccess)                                                     \
		{                                                                            \
			std::printf("CUDA error: %s  (%s:%d)\n", cudaGetErrorString(err_),        \
				__FILE__, __LINE__);                                                 \
			std::exit(1);                                                            \
		}                                                                            \
	} while (0)

// ---------------------------------------------------------------------------
// MNIST idx 文件读取（大端 32 位整数头）
// ---------------------------------------------------------------------------
static uint32_t read_be_u32(FILE* f)
{
	unsigned char b[4];
	if (std::fread(b, 1, 4, f) != 4)
	{
		std::printf("读取 idx 文件失败\n");
		std::exit(1);
	}
	return (static_cast<uint32_t>(b[0]) << 24) | (static_cast<uint32_t>(b[1]) << 16) |
		   (static_cast<uint32_t>(b[2]) << 8) | static_cast<uint32_t>(b[3]);
}

// 读图片：返回 [num][rows*cols] 的 float，已除以 255
static std::vector<float> load_images(const std::string& path, int& num, int& dim)
{
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f)
	{
		std::printf("打不开 %s\n", path.c_str());
		std::exit(1);
	}
	const uint32_t magic = read_be_u32(f);
	if (magic != 2051)
	{
		std::printf("%s 不是 image idx 文件（magic=%u）\n", path.c_str(), magic);
		std::exit(1);
	}
	const int n = static_cast<int>(read_be_u32(f));
	const int r = static_cast<int>(read_be_u32(f));
	const int c = static_cast<int>(read_be_u32(f));
	num = n;
	dim = r * c;

	std::vector<unsigned char> raw(static_cast<size_t>(n) * dim);
	if (std::fread(raw.data(), 1, raw.size(), f) != raw.size())
	{
		std::printf("%s 数据区不完整\n", path.c_str());
		std::exit(1);
	}
	std::fclose(f);

	std::vector<float> out(raw.size());
	for (size_t i = 0; i < raw.size(); ++i)
	{
		out[i] = static_cast<float>(raw[i]) / 255.0f;
	}
	return out;
}

static std::vector<int> load_labels(const std::string& path)
{
	FILE* f = std::fopen(path.c_str(), "rb");
	if (!f)
	{
		std::printf("打不开 %s\n", path.c_str());
		std::exit(1);
	}
	const uint32_t magic = read_be_u32(f);
	if (magic != 2049)
	{
		std::printf("%s 不是 label idx 文件（magic=%u）\n", path.c_str(), magic);
		std::exit(1);
	}
	const int n = static_cast<int>(read_be_u32(f));
	std::vector<unsigned char> raw(n);
	if (std::fread(raw.data(), 1, raw.size(), f) != raw.size())
	{
		std::printf("%s 数据区不完整\n", path.c_str());
		std::exit(1);
	}
	std::fclose(f);

	std::vector<int> out(n);
	for (int i = 0; i < n; ++i)
	{
		out[i] = static_cast<int>(raw[i]);
	}
	return out;
}

// ---------------------------------------------------------------------------
// GEMM：C[M,N] = op(A)[M,K] * op(B)[K,N]，全部行主序
//   TA=true 表示 A 在显存里按 (K,M) 存放（即需要转置），TB 同理按 (N,K) 存放。
// 32x32 分块 + 共享内存，每线程一个输出元素、一个 float 累加器。
// ---------------------------------------------------------------------------
// GEMM：C[M,N] = op(A)[M,K] * op(B)[K,N]，全部行主序
//   TA=true 表示 A 按 (K,M) 存放（要转置），TB=true 表示 B 按 (N,K) 存放。
//
// 64x64 分块、k 方向 16 的步长，每线程算 4x4 的小块：从共享内存搬 4+4 个数
// 就能做 16 次乘加，访存/计算比从 2:1 变成 1:2。比「一线程一元素」的 32x32
// 朴素分块快一个数量级，也比直接用 cuBLAS 更容易看清到底慢在哪一步。
// ---------------------------------------------------------------------------
template <bool TA, bool TB, int BM = 64, int BN = 64, int BK = 16, int TM = 4, int TN = 4>
__global__ void gemm_kernel(const float* __restrict__ A,
							const float* __restrict__ B,
							float* __restrict__ C,
							int M, int N, int K)
{
	constexpr int THREADS = (BM / TM) * (BN / TN);
	const int tx = threadIdx.x % (BN / TN);
	const int ty = threadIdx.x / (BN / TN);
	const int tid = threadIdx.x;

	__shared__ float As[BK][BM + 4];
	__shared__ float Bs[BK][BN + 4];

	const int row0 = blockIdx.y * BM + ty * TM;
	const int col0 = blockIdx.x * BN + tx * TN;

	float acc[TM][TN];
	for (int i = 0; i < TM; ++i)
	{
		for (int j = 0; j < TN; ++j)
		{
			acc[i][j] = 0.0f;
		}
	}

	for (int k0 = 0; k0 < K; k0 += BK)
	{
		for (int t = tid; t < BM * BK; t += THREADS)
		{
			const int r = t / BK;
			const int c = t % BK;
			const int gr = blockIdx.y * BM + r;
			const int gk = k0 + c;
			As[c][r] = (gr < M && gk < K)
				? (TA ? A[static_cast<size_t>(gk) * M + gr] : A[static_cast<size_t>(gr) * K + gk])
				: 0.0f;
		}
		for (int t = tid; t < BN * BK; t += THREADS)
		{
			const int c = t / BK;
			const int kk = t % BK;
			const int gc = blockIdx.x * BN + c;
			const int gk = k0 + kk;
			Bs[kk][c] = (gc < N && gk < K)
				? (TB ? B[static_cast<size_t>(gc) * K + gk] : B[static_cast<size_t>(gk) * N + gc])
				: 0.0f;
		}
		__syncthreads();

		for (int k = 0; k < BK; ++k)
		{
			float ra[TM];
			float rb[TN];
			for (int i = 0; i < TM; ++i)
			{
				ra[i] = As[k][ty * TM + i];
			}
			for (int j = 0; j < TN; ++j)
			{
				rb[j] = Bs[k][tx * TN + j];
			}
			for (int i = 0; i < TM; ++i)
			{
				for (int j = 0; j < TN; ++j)
				{
					acc[i][j] += ra[i] * rb[j];
				}
			}
		}
		__syncthreads();
	}

	for (int i = 0; i < TM; ++i)
	{
		const int r = row0 + i;
		if (r >= M)
		{
			continue;
		}
		for (int j = 0; j < TN; ++j)
		{
			const int c = col0 + j;
			if (c < N)
			{
				C[static_cast<size_t>(r) * N + c] = acc[i][j];
			}
		}
	}
}

template <bool TA, bool TB>
static void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
	constexpr int BM = 64;
	constexpr int BN = 64;
	dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
	gemm_kernel<TA, TB><<<grid, 256, 0, g_stream>>>(A, B, C, M, N, K);
}

// ---------------------------------------------------------------------------
// 逐元素 / 归约类 kernel
// ---------------------------------------------------------------------------
// z += b（按列广播），relu 可选
__global__ void bias_relu_kernel(float* z, const float* b, int rows, int cols, int do_relu)
{
	const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	const long long total = static_cast<long long>(rows) * cols;
	if (idx >= total)
	{
		return;
	}
	const int c = static_cast<int>(idx % cols);
	float v = z[idx] + b[c];
	z[idx] = do_relu ? fmaxf(v, 0.0f) : v;
}

// dZ *= (Z > 0)
__global__ void relu_mask_kernel(float* dz, const float* z, long long n)
{
	const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i < n && z[i] <= 0.0f)
	{
		dz[i] = 0.0f;
	}
}

// dst[c] += sum_r src[r][c]
__global__ void col_sum_acc_kernel(float* dst, const float* src, int rows, int cols)
{
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (c >= cols)
	{
		return;
	}
	float s = 0.0f;
	for (int r = 0; r < rows; ++r)
	{
		s += src[static_cast<size_t>(r) * cols + c];
	}
	dst[c] += s;
}

// p -= lr * g
__global__ void sgd_kernel(float* p, const float* g, long long n, float lr)
{
	const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i < n)
	{
		p[i] -= lr * g[i];
	}
}

// 按设备端保存的 shuffle 顺序一次性取出一个 batch 的图像和标签。
// 原来每步要往设备拷两次 idx/label，WDDM 下每次 pageable 拷贝要几百微秒，
// 比 kernel 本身还贵；现在每 epoch 只传一次顺序数组，取批全在 GPU 上做。
__global__ void gather_batch_kernel(const float* __restrict__ X, const int* __restrict__ dorder,
									const int* __restrict__ poff, const int* __restrict__ dlab,
									float* __restrict__ xb, int* __restrict__ lb,
									int rows, int cols)
{
	const int off = *poff;
	const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (i < static_cast<long long>(rows) * cols)
	{
		const int r = static_cast<int>(i / cols);
		const int c = static_cast<int>(i % cols);
		xb[i] = X[static_cast<size_t>(dorder[off + r]) * cols + c];
	}
	if (i < rows)
	{
		lb[i] = dlab[dorder[off + i]];
	}
}

// 取完这一批，把计数器推进一个 batch（设备端做，省掉一次主机侧参数变化）
__global__ void advance_off_kernel(int* poff, int step)
{
	*poff += step;
}

// softmax + 交叉熵：一个 block 处理一行（只有 thread 0 干活，10 类不值得并行）
//   loss[r] = -log p[label]，dz[r][c] = (p[c] - onehot[c]) * inv_batch
__global__ void softmax_ce_kernel(const float* logits, const int* label, float* dz,
								  float* loss_acc, int C, float inv_batch)
{
	if (threadIdx.x != 0)
	{
		return;
	}
	const int r = blockIdx.x;
	const float* z = logits + static_cast<size_t>(r) * C;
	float m = z[0];
	for (int c = 1; c < C; ++c)
	{
		m = fmaxf(m, z[c]);
	}
	float s = 0.0f;
	for (int c = 0; c < C; ++c)
	{
		s += expf(z[c] - m);
	}
	const int y = label[r];
	atomicAdd(loss_acc, -(z[y] - m) + logf(s));   // 每步只累加，不回拷主机

	float* g = dz + static_cast<size_t>(r) * C;
	for (int c = 0; c < C; ++c)
	{
		const float p = expf(z[c] - m) / s;
		g[c] = (p - (c == y ? 1.0f : 0.0f)) * inv_batch;
	}
}

// 评估：比较 argmax 与标签，命中则 correct + 1
__global__ void argmax_cmp_kernel(const float* logits, const int* label, int* correct, int C)
{
	if (threadIdx.x != 0)
	{
		return;
	}
	const int r = blockIdx.x;
	const float* z = logits + static_cast<size_t>(r) * C;
	int best = 0;
	for (int c = 1; c < C; ++c)
	{
		if (z[c] > z[best])
		{
			best = c;
		}
	}
	if (best == label[r])
	{
		atomicAdd(correct, 1);
	}
}

// ---------------------------------------------------------------------------
// 单独给 GEMM 记一次分：训练循环里 90% 的时间都在这两个大矩阵乘上，
// 混在一起看不出到底是 kernel 慢还是调度/拷贝慢。
// ---------------------------------------------------------------------------
template <bool TA>
static void bench_gemm(int M, int N, int K, int iters)
{
	const size_t na = static_cast<size_t>(M) * K;
	const size_t nb = static_cast<size_t>(K) * N;
	float *A = nullptr, *B = nullptr, *C = nullptr;
	CK(cudaMalloc(&A, na * sizeof(float)));
	CK(cudaMalloc(&B, nb * sizeof(float)));
	CK(cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(float)));
	CK(cudaMemset(A, 0, na * sizeof(float)));
	CK(cudaMemset(B, 0, nb * sizeof(float)));

	for (int i = 0; i < 5; ++i)
	{
		launch_gemm<TA, false>(A, B, C, M, N, K);
	}
	CK(cudaDeviceSynchronize());

	cudaEvent_t e0, e1;
	CK(cudaEventCreate(&e0));
	CK(cudaEventCreate(&e1));
	CK(cudaEventRecord(e0));
	for (int i = 0; i < iters; ++i)
	{
		launch_gemm<TA, false>(A, B, C, M, N, K);
	}
	CK(cudaEventRecord(e1));
	CK(cudaEventSynchronize(e1));
	float ms = 0.0f;
	CK(cudaEventElapsedTime(&ms, e0, e1));
	const double gflops = 2.0 * static_cast<double>(M) * N * K * iters / (ms * 1e6);
	std::printf("gemm TA=%d  M=%d N=%d K=%d  %.3f ms/次  %.1f GFLOPS\n", TA ? 1 : 0, M, N, K,
		ms / iters, gflops);
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
}

// ---------------------------------------------------------------------------
// 主机侧：模型与训练循环
// ---------------------------------------------------------------------------
struct Mlp
{
	int in = 784;
	int hid = 256;
	int cls = 10;
	int batch = 256;

	float *w1 = nullptr, *b1 = nullptr;
	float *w2 = nullptr, *b2 = nullptr;
	float *dw1 = nullptr, *db1 = nullptr;
	float *dw2 = nullptr, *db2 = nullptr;

	// 每步的中间量：xb 输入、z1 隐层线性输出、h 激活、z2 logits、dz 系列
	float *xb = nullptr, *z1 = nullptr, *h = nullptr, *z2 = nullptr;
	float *dz1 = nullptr, *dz2 = nullptr;
	float *loss = nullptr;   // 设备端 loss 累加器
	int* dorder = nullptr;   // 当前 epoch 的样本顺序（设备端）
	int* dlabel = nullptr;
	float* dbias_zero = nullptr;
};

static float* dev_alloc(size_t n)
{
	float* p = nullptr;
	CK(cudaMalloc(&p, n * sizeof(float)));
	CK(cudaMemset(p, 0, n * sizeof(float)));
	return p;
}

int main(int argc, char** argv)
{
	std::string data_dir = "../MMISTdataset";
	int epochs = 10;
	float lr = 0.3f;
	int batch = 256;

	for (int i = 1; i < argc; ++i)
	{
		if (!std::strcmp(argv[i], "--epochs") && i + 1 < argc) epochs = std::atoi(argv[++i]);
		else if (!std::strcmp(argv[i], "--lr") && i + 1 < argc) lr = static_cast<float>(std::atof(argv[++i]));
		else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
		else if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
	}

	if (argc > 1 && !std::strcmp(argv[1], "--bench-gemm"))
	{
		CK(cudaStreamCreate(&g_stream));
		CK(cudaStreamSynchronize(g_stream));
		bench_gemm<false>(1024, 256, 784, 200);   // 前向 X·W1
		bench_gemm<true>(784, 256, 1024, 200);    // 反向 dW1 = X^T·dZ1
		bench_gemm<false>(256, 256, 784, 200);    // batch=256 时的前向
		bench_gemm<false>(2048, 2048, 2048, 30);  // 大矩阵，看这张卡的实际上限
		return 0;
	}

	int dev = 0;
	cudaDeviceProp prop{};
	CK(cudaGetDeviceProperties(&prop, dev));
	CK(cudaSetDevice(dev));
	CK(cudaStreamCreate(&g_stream));
	std::printf("GPU: %s  SM %d.%d, %d SMs, %.1f GB\n", prop.name, prop.major, prop.minor,
		prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);

	int ntr = 0, dim = 0, nte = 0, dim_te = 0;
	const std::vector<float> xtr = load_images(data_dir + "/train-images-idx3-ubyte", ntr, dim);
	const std::vector<int> ytr = load_labels(data_dir + "/train-labels-idx1-ubyte");
	const std::vector<float> xte = load_images(data_dir + "/t10k-images-idx3-ubyte", nte, dim_te);
	const std::vector<int> yte = load_labels(data_dir + "/t10k-labels-idx1-ubyte");
	std::printf("train set %d x %d, test set %d x %d\n", ntr, dim, nte, dim_te);

	const int steps = ntr / batch;
	std::printf("batch=%d  lr=%.3f  epochs=%d  steps/epoch=%d\n", batch, lr, epochs, steps);

	Mlp m;
	m.batch = batch;
	m.in = dim;

	// 参数初始化：权重 uniform(-0.02, 0.02)，偏置 0（与 CPU 版 init_parameters 一致）
	std::mt19937 rng(20251104u);
	std::uniform_real_distribution<float> ud(-0.02f, 0.02f);
	std::vector<float> hw1(static_cast<size_t>(m.in) * m.hid), hw2(static_cast<size_t>(m.hid) * m.cls);
	for (float& v : hw1) v = ud(rng);
	for (float& v : hw2) v = ud(rng);

	const size_t n_w1 = static_cast<size_t>(m.in) * m.hid;
	m.w1 = dev_alloc(n_w1);
	m.w2 = dev_alloc(static_cast<size_t>(m.hid) * m.cls);
	m.b1 = dev_alloc(m.hid);
	m.b2 = dev_alloc(m.cls);
	m.dw1 = dev_alloc(n_w1);
	m.dw2 = dev_alloc(static_cast<size_t>(m.hid) * m.cls);
	m.db1 = dev_alloc(m.hid);
	m.db2 = dev_alloc(m.cls);
	CK(cudaMemcpy(m.w1, hw1.data(), n_w1 * sizeof(float), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(m.w2, hw2.data(), hw2.size() * sizeof(float), cudaMemcpyHostToDevice));

	m.xb = dev_alloc(static_cast<size_t>(batch) * m.in);
	m.z1 = dev_alloc(static_cast<size_t>(batch) * m.hid);
	m.h = dev_alloc(static_cast<size_t>(batch) * m.hid);
	m.z2 = dev_alloc(static_cast<size_t>(batch) * m.cls);
	m.dz1 = dev_alloc(static_cast<size_t>(batch) * m.hid);
	m.dz2 = dev_alloc(static_cast<size_t>(batch) * m.cls);
	m.loss = dev_alloc(batch);
	CK(cudaMalloc(&m.dorder, static_cast<size_t>(ntr) * sizeof(int)));
	CK(cudaMalloc(&m.dlabel, batch * sizeof(int)));

	float* dxtr = nullptr;
	CK(cudaMalloc(&dxtr, xtr.size() * sizeof(float)));
	CK(cudaMemcpy(dxtr, xtr.data(), xtr.size() * sizeof(float), cudaMemcpyHostToDevice));
	float* dxte = nullptr;
	CK(cudaMalloc(&dxte, xte.size() * sizeof(float)));
	CK(cudaMemcpy(dxte, xte.data(), xte.size() * sizeof(float), cudaMemcpyHostToDevice));
	int* dytrain = nullptr;
	int* dytest = nullptr;
	CK(cudaMalloc(&dytrain, ntr * sizeof(int)));
	CK(cudaMemcpy(dytrain, ytr.data(), ntr * sizeof(int), cudaMemcpyHostToDevice));
	CK(cudaMalloc(&dytest, nte * sizeof(int)));
	CK(cudaMemcpy(dytest, yte.data(), nte * sizeof(int), cudaMemcpyHostToDevice));

	std::vector<int> order(ntr);
	for (int i = 0; i < ntr; ++i) order[i] = i;

	int* doff = nullptr;
	CK(cudaMalloc(&doff, sizeof(int)));
	CK(cudaMemset(doff, 0, sizeof(int)));

	const int threads = 256;

	// 一步训练（取批 + 前向 + 反向 + 更新）的全部 kernel。
	// 「第几批」由设备端计数器 doff 决定，所以整步所有指针都是固定的，
	// 可以直接录成 CUDA Graph 反复回放 —— 这是小模型下最有效的优化：
	// 每步 11 次 launch 在 WDDM 上每次要花三四十微秒，比 kernel 本身还贵。
	auto launch_step = [&]() {
		gather_batch_kernel<<<(batch * m.in + threads - 1) / threads, threads, 0, g_stream>>>(
			dxtr, m.dorder, doff, dytrain, m.xb, m.dlabel, batch, m.in);
		advance_off_kernel<<<1, 1, 0, g_stream>>>(doff, batch);

		// --- 前向：X·W1 + b1 -> relu -> H·W2 + b2 ---
		launch_gemm<false, false>(m.xb, m.w1, m.z1, batch, m.hid, m.in);
		bias_relu_kernel<<<(batch * m.hid + threads - 1) / threads, threads, 0, g_stream>>>(
			m.z1, m.b1, batch, m.hid, 1);
		CK(cudaMemcpyAsync(m.h, m.z1, static_cast<size_t>(batch) * m.hid * sizeof(float),
			cudaMemcpyDeviceToDevice, g_stream));

		launch_gemm<false, false>(m.h, m.w2, m.z2, batch, m.cls, m.hid);
		bias_relu_kernel<<<(batch * m.cls + threads - 1) / threads, threads, 0, g_stream>>>(
			m.z2, m.b2, batch, m.cls, 0);

		softmax_ce_kernel<<<batch, 1, 0, g_stream>>>(m.z2, m.dlabel, m.dz2, m.loss, m.cls,
			1.0f / static_cast<float>(batch));

		// --- 反向 ---
		// 权重梯度由 GEMM 直接覆盖写，偏置梯度是累加，必须先清零（对应 CPU 的 zero_grad）
		CK(cudaMemsetAsync(m.db1, 0, m.hid * sizeof(float), g_stream));
		CK(cudaMemsetAsync(m.db2, 0, m.cls * sizeof(float), g_stream));

		launch_gemm<true, false>(m.h, m.dz2, m.dw2, m.hid, m.cls, batch);
		col_sum_acc_kernel<<<(m.cls + threads - 1) / threads, threads, 0, g_stream>>>(
			m.db2, m.dz2, batch, m.cls);

		launch_gemm<false, false>(m.dz2, m.w2, m.dz1, batch, m.hid, m.cls);
		relu_mask_kernel<<<(batch * m.hid + threads - 1) / threads, threads, 0, g_stream>>>(
			m.dz1, m.z1, static_cast<long long>(batch) * m.hid);

		launch_gemm<true, false>(m.xb, m.dz1, m.dw1, m.in, m.hid, batch);
		col_sum_acc_kernel<<<(m.hid + threads - 1) / threads, threads, 0, g_stream>>>(
			m.db1, m.dz1, batch, m.hid);

		// --- 更新 ---
		sgd_kernel<<<(static_cast<long long>(n_w1) + threads - 1) / threads, threads, 0, g_stream>>>(
			m.w1, m.dw1, n_w1, lr);
		sgd_kernel<<<(static_cast<long long>(m.hid) * m.cls + threads - 1) / threads, threads, 0, g_stream>>>(
			m.w2, m.dw2, static_cast<long long>(m.hid) * m.cls, lr);
		sgd_kernel<<<(m.hid + threads - 1) / threads, threads, 0, g_stream>>>(m.b1, m.db1, m.hid, lr);
		sgd_kernel<<<(m.cls + threads - 1) / threads, threads, 0, g_stream>>>(m.b2, m.db2, m.cls, lr);
	};

	launch_step();
	CK(cudaStreamSynchronize(g_stream));

	cudaGraph_t graph = nullptr;
	cudaGraphExec_t graph_exec = nullptr;
	CK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
	launch_step();
	CK(cudaStreamEndCapture(g_stream, &graph));
	CK(cudaGraphInstantiate(&graph_exec, graph, 0));

	const auto t_start = std::chrono::steady_clock::now();
	double train_ms = 0.0;

	for (int ep = 0; ep < epochs; ++ep)
	{
		std::shuffle(order.begin(), order.end(), rng);
		CK(cudaMemcpyAsync(m.dorder, order.data(), static_cast<size_t>(ntr) * sizeof(int),
			cudaMemcpyHostToDevice, g_stream));
		CK(cudaMemsetAsync(doff, 0, sizeof(int), g_stream));
		CK(cudaMemsetAsync(m.loss, 0, sizeof(float), g_stream));
		CK(cudaStreamSynchronize(g_stream));

		const auto t0 = std::chrono::steady_clock::now();
		for (int s = 0; s < steps; ++s)
		{
			CK(cudaGraphLaunch(graph_exec, g_stream));
		}
		float epoch_loss = 0.0f;
		CK(cudaMemcpyAsync(&epoch_loss, m.loss, sizeof(float), cudaMemcpyDeviceToHost, g_stream));
		CK(cudaStreamSynchronize(g_stream));

		const auto t1 = std::chrono::steady_clock::now();
		const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		train_ms += ms;
		std::printf("epoch %2d  loss %.6f  %.1f ms\n", ep + 1,
			epoch_loss / static_cast<double>(steps) / batch, ms);
	}
	cudaGraphExecDestroy(graph_exec);
	cudaGraphDestroy(graph);

	// ---- 评估（前向 only，按 batch 分块）----
	auto evaluate = [&](const float* dx, const int* dy, int n, const char* tag) {
		int* dcorr = nullptr;
		CK(cudaMalloc(&dcorr, sizeof(int)));
		CK(cudaMemset(dcorr, 0, sizeof(int)));
		float* zb = dev_alloc(static_cast<size_t>(batch) * m.cls);
		float* hb = dev_alloc(static_cast<size_t>(batch) * m.hid);
		float* z1b = dev_alloc(static_cast<size_t>(batch) * m.hid);
		float* xb = dev_alloc(static_cast<size_t>(batch) * m.in);
		for (int off = 0; off + batch <= n; off += batch)
		{
			CK(cudaMemcpyAsync(xb, dx + static_cast<size_t>(off) * m.in,
				static_cast<size_t>(batch) * m.in * sizeof(float), cudaMemcpyDeviceToDevice));
			launch_gemm<false, false>(xb, m.w1, z1b, batch, m.hid, m.in);
			bias_relu_kernel<<<(batch * m.hid + threads - 1) / threads, threads>>>(
				z1b, m.b1, batch, m.hid, 1);
			CK(cudaMemcpyAsync(hb, z1b, static_cast<size_t>(batch) * m.hid * sizeof(float),
				cudaMemcpyDeviceToDevice));
			launch_gemm<false, false>(hb, m.w2, zb, batch, m.cls, m.hid);
			bias_relu_kernel<<<(batch * m.cls + threads - 1) / threads, threads>>>(
				zb, m.b2, batch, m.cls, 0);
			argmax_cmp_kernel<<<batch, 1>>>(zb, dy + off, dcorr, m.cls);
		}
		CK(cudaDeviceSynchronize());
		int corr = 0;
		CK(cudaMemcpy(&corr, dcorr, sizeof(int), cudaMemcpyDeviceToHost));
		std::printf("%s acc: %.4f\n", tag, static_cast<float>(corr) / static_cast<float>(n));
		cudaFree(dcorr);
		cudaFree(zb);
		cudaFree(hb);
		cudaFree(z1b);
		cudaFree(xb);
	};
	evaluate(dxtr, dytrain, ntr, "train");
	evaluate(dxte, dytest, nte, "test");

	const auto t_end = std::chrono::steady_clock::now();
	const double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	const double macs = static_cast<double>(ntr) * epochs * 3.0 *
						(static_cast<double>(m.in) * m.hid + static_cast<double>(m.hid) * m.cls);
	std::printf("train time %.1f ms, total %.1f ms (incl. data load and eval)\n", train_ms, total_ms);
	std::printf("effective %.1f GFLOPS (assuming 3x MACs for fwd+bwd)\n", 2.0 * macs / (train_ms * 1e6));
	return 0;
}
