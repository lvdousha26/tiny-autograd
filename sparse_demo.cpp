// 稀疏能力的梯度校验：
//   1) 新增的 TOPK 算子（前向每行恰好 k 个非零 + 中心差分梯度）
//   2) Net::set_sparse_penalty 往隐层输出注入的 L1 梯度
// 这两个是 Python 包训练 SAE 时全部「非标准」的部分，所以单独验一遍。
#include <algorithm>
#include <chrono>
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

static void check_dynamic_batch()
{
	const int in = 4;
	const int hidden = 6;
	const int k = 3;

	auto build = [&](int rows)
	{
		Net* n = new Net(rows);
		n->add_layer(in, hidden, FCLayer::TOPK, k);
		n->add_layer(hidden, in, FCLayer::NONE);
		n->set_loss_function(Net::MEAN_SQUARED);
		n->set_sparse_penalty(0.05f, 0);
		return n;
	};
	auto rows_of = [](int rows, int cols)
	{
		std::vector<std::vector<float>> v(rows, std::vector<float>(cols));
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				v[r][c] = 0.3f * (r + 1) - 0.1f * (c + 1);
			}
		}
		return v;
	};

	Net* net = build(4);
	net->init_parameters(0.0f, 0.3f);
	const auto x4 = rows_of(4, in);
	const auto y4 = rows_of(4, in);
	net->train_batch(x4, y4);          // 先训练一步，让参数不再是初值

	const auto w0 = net->layer_weights(0)->get_data();
	const auto w1 = net->layer_weights(1)->get_data();
	const auto b0 = net->layer_bias(0)->get_data();
	const float alpha_before = net->layer_outputs(0)->get_alpha();

	printf("\n--- dynamic batch (rebuild graph, keep params) ---\n");
	net->resize_batch(7);
	printf("graph rows 4 -> %d\n", net->get_batch_rows());
	bool same = true;
	const auto& nw0 = net->layer_weights(0)->get_data();
	const auto& nw1 = net->layer_weights(1)->get_data();
	const auto& nb0 = net->layer_bias(0)->get_data();
	for (int i = 0; i < static_cast<int>(w0.size()); ++i)
	{
		for (int j = 0; j < static_cast<int>(w0[i].size()); ++j)
		{
			same &= (w0[i][j] == nw0[i][j]);
		}
	}
	for (int i = 0; i < static_cast<int>(w1.size()); ++i)
	{
		for (int j = 0; j < static_cast<int>(w1[i].size()); ++j)
		{
			same &= (w1[i][j] == nw1[i][j]);
		}
	}
	for (int i = 0; i < static_cast<int>(b0.size()); ++i)
	{
		for (int j = 0; j < static_cast<int>(b0[i].size()); ++j)
		{
			same &= (b0[i][j] == nb0[i][j]);
		}
	}
	same &= (alpha_before == net->layer_outputs(0)->get_alpha());
	printf("params / TopK k preserved after rebuild: %s\n", same ? "yes" : "NO");

	// 用 7 行训练一步，再缩回 4 行，参数必须带着这步的更新
	const auto x7 = rows_of(7, in);
	const auto y7 = rows_of(7, in);
	net->train_batch(x7, y7);
	net->resize_batch(3);
	const auto x3 = rows_of(3, in);
	const auto y3 = rows_of(3, in);
	const float loss_dynamic = net->train_batch(x3, y3);
	const auto grad_dynamic = net->layer_weights(0)->get_grad();

	// 同参数、原生 3 行图，梯度应逐元素一致
	Net* fresh = build(3);
	fresh->layer_weights(0)->get_data() = net->layer_weights(0)->get_data();
	fresh->layer_weights(1)->get_data() = net->layer_weights(1)->get_data();
	fresh->layer_bias(0)->get_data() = net->layer_bias(0)->get_data();
	fresh->layer_bias(1)->get_data() = net->layer_bias(1)->get_data();
	const float loss_fresh = fresh->train_batch(x3, y3);
	const auto& grad_fresh = fresh->layer_weights(0)->get_grad();
	double worst = 0.0;
	for (int i = 0; i < static_cast<int>(grad_dynamic.size()); ++i)
	{
		for (int j = 0; j < static_cast<int>(grad_dynamic[i].size()); ++j)
		{
			worst = std::max(worst, static_cast<double>(
				std::fabs(grad_dynamic[i][j] - grad_fresh[i][j])));
		}
	}
	printf("loss: rebuilt %.6f vs native-3-row %.6f\n", loss_dynamic, loss_fresh);
	printf("dL/dW max diff between rebuilt graph and native graph = %.3e\n", worst);

	// 重建开销：呼应 graph_demo 里 rebuild vs reuse 的 7.4x
	const int iters = 2000;
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; ++i)
	{
		net->resize_batch(3 + (i & 1));
	}
	auto t1 = std::chrono::steady_clock::now();
	const double per_resize = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
	net->resize_batch(2);
	const auto x2 = rows_of(2, in);
	const auto y2 = rows_of(2, in);
	auto t2 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; ++i)
	{
		net->train_batch(x2, y2);
	}
	auto t3 = std::chrono::steady_clock::now();
	const double per_step = std::chrono::duration<double, std::micro>(t3 - t2).count() / iters;
	printf("resize %.3f us  vs  one train step %.3f us  (%.1fx)\n",
		per_resize, per_step, per_resize / per_step);

	delete fresh;
	delete net;
}

int main()
{
	rand_generator().seed(2026u);   // 参数初始化可复现
	check_topk();
	check_sparse_penalty();
	check_dynamic_batch();
	return 0;
}
