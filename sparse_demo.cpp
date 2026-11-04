// 稀疏能力的梯度校验：
//   1) 新增的 TOPK 算子（前向每行恰好 k 个非零 + 中心差分梯度）
//   2) Net::set_sparse_penalty 往隐层输出注入的 L1 梯度
// 这两个是 Python 包训练 SAE 时全部「非标准」的部分，所以单独验一遍。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "Graph.h"
#include "Net.h"
#include "Tensor.h"
#include "utils.h"

using CF = Tensor::CalculateFun;

static void check_topk()
{
	const int rows = 3;
	const int cols = 6;
	const int k = 2;
	Tensor x(rows, cols, true);
	const float base[3][6] = {
		{ -0.4f, 1.7f, 0.2f, 2.9f, 0.8f, -1.0f },
		{ 3.1f, -0.2f, 0.9f, 0.1f, 2.4f, 0.6f },
		{ 0.5f, 1.1f, -0.7f, 2.2f, 0.3f, 1.9f }
	};
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			x.get_data()[r][c] = base[r][c];
		}
	}

	Tensor act(rows, cols, true);
	act.set_left(&x);
	act.set_calculate_fn(CF::TOPK);
	act.set_alpha(static_cast<float>(k));

	Tensor target(rows, cols);
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			target.get_data()[r][c] = 0.3f;
		}
	}

	Tensor loss(1, 1, true);
	loss.set_left(&act);
	loss.set_right(&target);
	loss.set_calculate_fn(CF::VECTOR_SQUARED);

	auto forward_all = [&]()
	{
		x.forward_calculate_data();
		act.forward_calculate_data();
		loss.forward_calculate_data();
	};

	forward_all();
	printf("--- TOPK forward (k=%d) ---\n", k);
	for (int r = 0; r < rows; ++r)
	{
		int live = 0;
		printf("  row %d: ", r);
		for (int c = 0; c < cols; ++c)
		{
			printf("%5.2f ", act.get_data()[r][c]);
			live += (act.get_data()[r][c] > 0.0f) ? 1 : 0;
		}
		printf(" live=%d\n", live);
		if (live != k)
		{
			printf("  ERROR: expected exactly %d live features\n", k);
		}
	}

	x.zero_grad();
	act.zero_grad();
	loss.zero_grad();
	loss.one_grad();
	graph::backward_memo(&loss, { &x });

	const float h = 1e-3f;
	double worst = 0.0;
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			const float orig = x.get_data()[r][c];
			x.get_data()[r][c] = orig + h;
			forward_all();
			const float lp = loss.get_data()[0][0];
			x.get_data()[r][c] = orig - h;
			forward_all();
			const float lm = loss.get_data()[0][0];
			x.get_data()[r][c] = orig;
			const float numeric = (lp - lm) / (2.0f * h);
			worst = std::max(worst, static_cast<double>(
				std::fabs(numeric - x.get_grad()[r][c])));
		}
	}
	forward_all();
	printf("TOPK dL/dx : max |analytic - numeric| = %.3e\n", worst);
}

static void check_sparse_penalty()
{
	const int rows = 3;
	const int in = 3;
	const int hidden = 5;
	const float lambda = 0.15f;

	Net net(rows);
	net.add_layer(in, hidden, FCLayer::RELU);
	net.add_layer(hidden, in, FCLayer::NONE);
	net.set_loss_function(Net::MEAN_SQUARED);
	net.set_sparse_penalty(lambda, 0);
	net.init_parameters(0.0f, 0.35f);

	std::vector<std::vector<float>> xs(rows, std::vector<float>(in));
	std::vector<std::vector<float>> ys(rows, std::vector<float>(in));
	const float in_vals[3][3] = {
		{ 1.0f, -0.5f, 0.3f }, { -0.2f, 0.8f, 1.1f }, { 0.6f, 0.4f, -0.9f }
	};
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < in; ++c)
		{
			xs[r][c] = in_vals[r][c];
			ys[r][c] = 0.25f;
		}
	}

	// 稀疏项只在 train_batch 里加进返回的 loss；数值梯度要自己补上这一项
	auto total_loss = [&]()
	{
		float l = net.forward_batch(xs, ys);
		const Tensor* z = net.layer_outputs(0);
		float s = 0.0f;
		for (int r = 0; r < z->get_row_dimension(); ++r)
		{
			for (int c = 0; c < z->get_col_dimension(); ++c)
			{
				if (z->get_data()[r][c] > 0.0f)
				{
					s += z->get_data()[r][c];
				}
			}
		}
		return l + lambda * s / static_cast<float>(rows);
	};

	net.zero_grad();
	const float analytic_loss = net.train_batch(xs, ys);
	Tensor* w = net.layer_weights(0);
	std::vector<std::vector<float>> analytic = w->get_grad();

	const float h = 1e-3f;
	double worst = 0.0;
	for (int i = 0; i < w->get_row_dimension(); ++i)
	{
		for (int j = 0; j < w->get_col_dimension(); ++j)
		{
			const float orig = w->get_data()[i][j];
			w->get_data()[i][j] = orig + h;
			const float lp = total_loss();
			w->get_data()[i][j] = orig - h;
			const float lm = total_loss();
			w->get_data()[i][j] = orig;
			const float numeric = (lp - lm) / (2.0f * h);
			worst = std::max(worst, static_cast<double>(
				std::fabs(numeric - analytic[i][j])));
		}
	}
	printf("--- sparse L1 penalty ---\n");
	printf("train_batch loss = %.6f, recomputed = %.6f\n", analytic_loss, total_loss());
	printf("dL/dW_enc : max |analytic - numeric| = %.3e\n", worst);
}

int main()
{
	rand_generator().seed(2026u);   // 参数初始化可复现
	check_topk();
	check_sparse_penalty();
	return 0;
}
