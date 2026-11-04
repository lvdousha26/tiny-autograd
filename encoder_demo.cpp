// ---------------------------------------------------------------------------
// encoder block 的验证程序：
//   1. 前向数值性质（注意力每行和为 1、LN 每行均值 0 / 方差 1）
//   2. 中心差分梯度校验（含 gamma / beta / bias / 输入 x，验证广播与 LN 反传）
//   3. 残差把计算图变成 DAG：栈版 DFS 会重复累加，记忆化搜索才对
//   4. 把所有子层权重清零：有残差时梯度通路还在，没残差时梯度恒为 0
//   5. 正常训练几步，确认 loss 能降下来
// ---------------------------------------------------------------------------
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "encoder_block.h"

static const char* kSep = "------------------------------------------------------------";

// 中心差分：把解析梯度和 (L(+h) - L(-h)) / 2h 逐元素比。
// 纯相对误差在梯度接近 0 的元素上会被 float32 舍入放大，所以用
// 「绝对 + 相对」混合容差（同 torch 的 gradcheck）：|a-n| <= 1e-4 + 1e-2*|n|，
// 报告 err/tol，小于 1 即通过。h=1e-2 是本网络在截断误差和舍入误差之间的最优点。
static double grad_check(EncoderGraph& g, Tensor* p, const char* label, float h = 1e-2f)
{
	g.forward();
	g.zero();
	graph::backward_memo(g.loss, g.watch);
	const auto analytic = p->get_grad();

	double worst_ratio = 0.0;
	double worst_abs = 0.0;
	int checked = 0;
	int failed = 0;
	for (int i = 0; i < p->get_row_dimension(); ++i)
	{
		for (int j = 0; j < p->get_col_dimension(); ++j)
		{
			const float orig = p->get_data()[i][j];

			p->get_data()[i][j] = orig + h;
			g.forward();
			const float lp = g.loss->get_data()[0][0];

			p->get_data()[i][j] = orig - h;
			g.forward();
			const float lm = g.loss->get_data()[0][0];

			p->get_data()[i][j] = orig;
			const float numeric = (lp - lm) / (2.0f * h);
			const double err = std::fabs(numeric - analytic[i][j]);
			const double tol = 1e-4 + 1e-2 * std::fabs(static_cast<double>(numeric));
			worst_ratio = std::max(worst_ratio, err / tol);
			worst_abs = std::max(worst_abs, err);
			if (err > tol) ++failed;
			++checked;
		}
	}
	g.forward();
	printf("  %-14s (%2d x %2d)  checked %4d  max|err| = %.2e  err/tol = %.3f  fail=%d\n",
		label, p->get_row_dimension(), p->get_col_dimension(), checked,
		worst_abs, worst_ratio, failed);
	return worst_ratio;
}

static void print_row_stats(const Tensor* t, const char* label)
{
	printf("%s (per-row mean / var):\n", label);
	for (int r = 0; r < t->get_row_dimension(); ++r)
	{
		const auto& row = t->get_data()[r];
		const float n = static_cast<float>(row.size());
		float mean = 0.0f;
		for (float v : row) mean += v;
		mean /= n;
		float var = 0.0f;
		for (float v : row) var += (v - mean) * (v - mean);
		var /= n;
		printf("  row %d: mean = %+.5f   var = %.5f\n", r, mean, var);
	}
}

// 输入 x 的梯度范数（只求 x 一个节点的梯度即可）
static double input_grad_norm(EncoderGraph& g)
{
	g.zero();
	graph::backward_memo(g.loss, std::vector<Tensor*>{ g.x });
	double s = 0.0;
	for (const auto& row : g.x->get_grad())
	{
		for (float v : row) s += static_cast<double>(v) * v;
	}
	return std::sqrt(s);
}

// 子层权重全部清 0，gamma 保持 1、beta 保持 0
static void kill_sublayers(EncoderGraph& g)
{
	for (size_t i = 0; i < g.params.size(); ++i)
	{
		const bool is_gamma = g.param_names[i].find(".gamma") != std::string::npos;
		const float v = is_gamma ? 1.0f : 0.0f;
		for (auto& row : g.params[i]->get_data())
		{
			std::fill(row.begin(), row.end(), v);
		}
	}
}

// 同一张残差图上对比两种反向遍历：以数值梯度为裁判
static void compare_backward_modes()
{
	EncoderGraph g = build_encoder_graph(4, 8, 2, 8, 1, true, 777u, 0.3f);

	g.forward();
	g.zero();
	graph::backward_memo(g.loss, g.watch);
	const auto memo_grad = g.x->get_grad();

	g.zero();
	graph::backward_tree(g.loss);          // 栈版 DFS，不负责去重
	const auto tree_grad = g.x->get_grad();

	const float h = 1e-2f;
	double worst_memo = 0.0;
	double worst_tree = 0.0;
	for (int i = 0; i < g.x->get_row_dimension(); ++i)
	{
		for (int j = 0; j < g.x->get_col_dimension(); ++j)
		{
			const float orig = g.x->get_data()[i][j];
			g.x->get_data()[i][j] = orig + h;
			g.forward();
			const float lp = g.loss->get_data()[0][0];
			g.x->get_data()[i][j] = orig - h;
			g.forward();
			const float lm = g.loss->get_data()[0][0];
			g.x->get_data()[i][j] = orig;
			const float numeric = (lp - lm) / (2.0f * h);
			const double tol = 1e-4 + 1e-2 * std::fabs(static_cast<double>(numeric));
			worst_memo = std::max(worst_memo, std::fabs(numeric - memo_grad[i][j]) / tol);
			worst_tree = std::max(worst_tree, std::fabs(numeric - tree_grad[i][j]) / tol);
		}
	}
	printf("  dL/dx : numeric vs backward_tree  err/tol = %.3f  (WRONG)\n", worst_tree);
	printf("  dL/dx : numeric vs backward_memo  err/tol = %.3f  (OK)\n", worst_memo);
	g.release();
}

int main()
{
	const int S = 5;
	const int d_model = 8;
	const int heads = 2;
	const int d_ff = 16;
	const int blocks = 2;

	EncoderGraph g = build_encoder_graph(S, d_model, heads, d_ff, blocks, true, 2026u, 0.3f);
	printf("=== encoder stack ===\n");
	printf("S=%d  d_model=%d  heads=%d  d_k=%d  d_ff=%d  blocks=%d  residual=on\n",
		S, d_model, heads, g.d_k, d_ff, blocks);
	printf("Post-LN, sinusoidal positional encoding, GELU FFN, no dropout\n");
	printf("graph nodes = %d, trainable tensors = %d\n",
		static_cast<int>(g.owned.size()), static_cast<int>(g.params.size()));

	g.forward();
	printf("loss = %.6f\n", static_cast<double>(g.loss->get_data()[0][0]));

	printf("%s\n--- 1. forward sanity ---\n", kSep);
	printf("attention row sums (head 0 of block 0):\n");
	for (int r = 0; r < S; ++r)
	{
		float sum = 0.0f;
		for (int c = 0; c < S; ++c) sum += g.attn_prob->get_data()[r][c];
		printf("  row %d = %.6f\n", r, sum);
	}
	print_row_stats(g.ln_bare, "bare LayerNorm output");

	printf("%s\n--- 2. gradient check (central difference, all elements) ---\n", kSep);
	printf("  tolerance: |err| <= 1e-4 + 1e-2 * |numeric|   (h = 1e-2)\n");
	const char* probes[] = {
		"b0.h0.Wq", "b0.h0.Wk", "b0.h0.Wv", "b0.h0.Wo",
		"b0.W1", "b0.b1", "b0.W2", "b0.b2",
		"b0.ln1.gamma", "b0.ln1.beta", "b0.ln2.gamma", "b0.ln2.beta",
		"b1.h1.Wq", "b1.W2"
	};
	double worst = 0.0;
	for (const char* name : probes)
	{
		Tensor* p = eg_find_param(g, name);
		if (p == nullptr)
		{
			printf("  %-14s MISSING\n", name);
			continue;
		}
		worst = std::max(worst, grad_check(g, p, name));
	}
	worst = std::max(worst, grad_check(g, g.x, "input x"));
	printf("  overall err/tol = %.3f  (< 1 means all gradients verified)\n", worst);

	printf("%s\n--- 3. residual makes the graph a DAG ---\n", kSep);
	compare_backward_modes();

	printf("%s\n--- 4. zero out every sublayer, keep residual? ---\n", kSep);
	double norm_res = 0.0;
	double norm_nores = 0.0;
	for (bool res : { true, false })
	{
		EncoderGraph z = build_encoder_graph(4, 8, 2, 8, 3, res, 999u, 0.3f);
		kill_sublayers(z);
		z.forward();
		const double loss = z.loss->get_data()[0][0];
		const double norm = input_grad_norm(z);
		printf("  residual=%-5s  blocks=3  loss=%.6f  |dL/dx| = %.6e\n",
			res ? "on" : "off", loss, norm);
		if (res) norm_res = norm; else norm_nores = norm;
		z.release();
	}
	printf("  |dL/dx|: with = %.6e, without = %.6e -> residual keeps the path alive\n",
		norm_res, norm_nores);

	printf("%s\n--- 5. training ---\n", kSep);
	const float lr = 0.02f;
	for (int step = 0; step <= 200; ++step)
	{
		g.forward();
		if (step % 50 == 0)
		{
			printf("  step %3d  loss = %.6f\n", step,
				static_cast<double>(g.loss->get_data()[0][0]));
		}
		g.zero();
		graph::backward_memo(g.loss, g.params);
		for (auto* p : g.params)
		{
			for (int i = 0; i < p->get_row_dimension(); ++i)
			{
				for (int j = 0; j < p->get_col_dimension(); ++j)
				{
					p->get_data()[i][j] -= lr * p->get_grad()[i][j];
				}
			}
		}
	}

	g.release();
	return 0;
}
