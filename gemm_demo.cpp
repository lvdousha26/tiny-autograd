// 逐样本 matvec 与批量 GEMM 的对照，用于判断上 GPU 是否划算
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const int IN = 784;
static const int OUT = 256;
static const int TOTAL = 8192;

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0)
{
	return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// 现行做法：一次一个样本，y = W * x
static void matvec(const std::vector<float>& W, const float* x, float* y)
{
	for (int i = 0; i < OUT; ++i)
	{
		const float* row = &W[i * IN];
		float sum = 0.0f;
		for (int j = 0; j < IN; ++j)
		{
			sum += row[j] * x[j];
		}
		y[i] = sum;
	}
}

// 批量：Y = X * W^T，X 为 B x IN，结果按 OUT x B 存放
static void gemm_plain(const std::vector<float>& W, const float* X, std::vector<float>& Y, int B)
{
	for (int b = 0; b < B; ++b)
	{
		const float* x = &X[b * IN];
		for (int i = 0; i < OUT; ++i)
		{
			const float* row = &W[i * IN];
			float sum = 0.0f;
			for (int j = 0; j < IN; ++j)
			{
				sum += row[j] * x[j];
			}
			Y[i * B + b] = sum;
		}
	}
}

// 批量 + 线程并行：一行权重服务整个 batch，写回连续内存
static void gemm_omp(const std::vector<float>& W, const float* X, std::vector<float>& Y, int B)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < OUT; ++i)
	{
		const float* row = &W[i * IN];
		for (int b = 0; b < B; ++b)
		{
			const float* x = &X[b * IN];
			float sum = 0.0f;
			for (int j = 0; j < IN; ++j)
			{
				sum += row[j] * x[j];
			}
			Y[i * B + b] = sum;
		}
	}
}

int main()
{
	std::mt19937 gen(42);
	std::normal_distribution<float> dist(0.0f, 0.02f);

	std::vector<float> W(OUT * IN);
	for (auto& v : W)
	{
		v = dist(gen);
	}
	std::vector<float> X(TOTAL * IN);
	for (size_t i = 0; i < X.size(); ++i)
	{
		X[i] = static_cast<float>(i % 255) / 255.0f;
	}
	std::vector<float> Y(TOTAL * OUT, 0.0f);

	const double flop_per_sample = 2.0 * IN * OUT;
	const double total_flop = TOTAL * flop_per_sample;

	printf("workload: %d samples x %d x %d matvec\n\n", TOTAL, IN, OUT);

	{
		auto t0 = Clock::now();
		for (int b = 0; b < TOTAL; ++b)
		{
			matvec(W, &X[b * IN], &Y[b * OUT]);
		}
		double ms = ms_since(t0);
		printf("per-sample matvec   : %8.2f ms  %6.1f us/sample  %6.2f GFLOPS\n",
			   ms, ms * 1000.0 / TOTAL, total_flop / ms / 1e6);
	}

	const int batches[] = { 1, 8, 32, 128, 256 };

	for (int B : batches)
	{
		auto t0 = Clock::now();
		for (int off = 0; off + B <= TOTAL; off += B)
		{
			gemm_plain(W, &X[off * IN], Y, B);
		}
		double ms = ms_since(t0);
		printf("gemm batch=%-4d     : %8.2f ms  %6.1f us/sample  %6.2f GFLOPS\n",
			   B, ms, ms * 1000.0 / TOTAL, total_flop / ms / 1e6);
	}

	for (int B : batches)
	{
		auto t0 = Clock::now();
		for (int off = 0; off + B <= TOTAL; off += B)
		{
		  gemm_omp(W, &X[off * IN], Y, B);
		}
		double ms = ms_since(t0);
		printf("gemm+omp batch=%-4d : %8.2f ms  %6.1f us/sample  %6.2f GFLOPS\n",
			   B, ms, ms * 1000.0 / TOTAL, total_flop / ms / 1e6);
	}

	const double gpu_peak = 13000.0;
	const double peak_us = total_flop / gpu_peak / 1e9 * 1e6;
	printf("\nGPU reference (RTX 4060 Laptop, ~13 TFLOPS fp32 peak, NOT measured):\n");
	printf("  compute only      : %6.2f us/sample (peak bound)\n", peak_us / TOTAL);
	printf("  at 50%% efficiency : %6.2f us/sample\n", peak_us * 2.0 / TOTAL);
	printf("  plus kernel launch and host-device copies, which dominate at batch=1\n");
	return 0;
}
