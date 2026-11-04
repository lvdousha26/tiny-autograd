// ---------------------------------------------------------------------------
// 真正的 Transformer encoder block：多头自注意力 + 残差 + LayerNorm + GELU FFN。
//
// 关键设计（绕开 Tensor 只有 left/right 两个输入的限制）：
//   LayerNorm 看起来需要 x / gamma / beta 三个输入，但把它拆成三个二元/一元节点
//   就完全不用动 Tensor 的结构：
//       ln0    = LAYER_NORM(x)              单目，逐行归一化、不含仿射
//       scaled = ln0 * gamma                gamma 是 (1, D)，靠 row_of 行广播
//       out    = scaled + beta              beta  是 (1, D)，同上
//   而 VECTOR_MUL_VECTOR / VECTOR_ADD_VECTOR 的右梯度本来就会对所有行累加到
//   grad[0] 上，所以 gamma/beta 的梯度自动是对的。
//
// 多头注意力没有拼接算子，用的是等价的代数分解：
//       concat(a_1..a_H) @ Wo  ==  sum_h a_h @ Wo_h
//   （Wo 按行切成 H 块）。于是每个头独立算 q/k/v/softmax，最后把投影结果相加。
//
// 残差让计算图变成 DAG —— 这正是必须用 graph::backward_memo 而不是栈版 DFS 的场景。
//
// 这里用的是原论文「Attention Is All You Need」的 Post-LN 结构：LN(x + sublayer(x))，
// 不是后来常见的 Pre-LN。输入先加正弦位置编码，否则自注意力对词序不敏感。
// dropout 有意省略：它是随机的，会让中心差分梯度校验失去意义。
// ---------------------------------------------------------------------------
#ifndef _ENCODER_BLOCK_H_
#define _ENCODER_BLOCK_H_

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "Graph.h"
#include "Tensor.h"

using CF = Tensor::CalculateFun;

struct EncoderGraph
{
	int S = 0;             // 序列长度
	int d_model = 0;
	int heads = 0;
	int d_k = 0;           // 每头维度 = d_model / heads
	int d_ff = 0;
	int blocks = 0;
	bool residual = true;

	std::vector<Tensor*> owned;        // 拓扑序：前向按这个顺序跑
	std::vector<Tensor*> params;       // 可训练叶子
	std::vector<std::string> param_names;
	std::vector<Tensor*> watch;        // params + 输入 x，梯度校验用

	Tensor* x = nullptr;
	Tensor* pos = nullptr;         // 正弦位置编码（常量叶子）
	Tensor* target = nullptr;
	Tensor* loss = nullptr;
	Tensor* output = nullptr;

	// 第一个 block 的几个中间量，用于在 demo 里验证数值性质
	Tensor* attn_prob = nullptr;       // 注意力矩阵，每行和应为 1
	Tensor* ln_bare = nullptr;         // 无仿射的 LN 输出，每行均值 0 / 方差 1
	Tensor* ln1_out = nullptr;         // 带 gamma/beta 的 LN1 输出

	void forward()
	{
		for (auto* t : owned)
		{
			t->forward_calculate_data();
		}
	}

	void zero()
	{
		for (auto* t : owned)
		{
			t->zero_grad();
		}
	}

	void release()
	{
		for (auto* t : owned)
		{
			delete t;
		}
		owned.clear();
		params.clear();
		watch.clear();
	}
};

// 建一个叶子：stddev > 0 时用正态分布初始化，否则填成 fixed。
// trainable 决定是否放进优化器参数表（x / target / 常量只是占位叶子）。
static Tensor* eg_make(EncoderGraph& g, int rows, int cols, const std::string& name,
					   std::mt19937& gen, float stddev, float fixed, bool trainable)
{
	auto* t = new Tensor(rows, cols, true);
	if (stddev > 0.0f)
	{
		std::normal_distribution<float> nd(0.0f, stddev);
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				t->get_data()[i][j] = nd(gen);
			}
		}
	}
	else
	{
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				t->get_data()[i][j] = fixed;
			}
		}
	}
	g.owned.push_back(t);
	if (trainable)
	{
		g.params.push_back(t);
		g.param_names.push_back(name);
	}
	return t;
}

static Tensor* eg_unary(EncoderGraph& g, int rows, int cols, Tensor* l, CF fn)
{
	auto* t = new Tensor(rows, cols, true);
	t->set_left(l);
	t->set_calculate_fn(fn);
	g.owned.push_back(t);
	return t;
}

static Tensor* eg_binary(EncoderGraph& g, int rows, int cols, Tensor* l, Tensor* r, CF fn)
{
	auto* t = new Tensor(rows, cols, true);
	t->set_left(l);
	t->set_right(r);
	t->set_calculate_fn(fn);
	g.owned.push_back(t);
	return t;
}

// LN(x) * gamma + beta，gamma 初值 1、beta 初值 0（标准做法）
static Tensor* eg_layer_norm(EncoderGraph& g, Tensor* x, int S, int D,
							 const std::string& tag, std::mt19937& gen,
							 Tensor*& bare_out)
{
	auto* ln0 = eg_unary(g, S, D, x, CF::LAYER_NORM);
	if (bare_out == nullptr)
	{
		bare_out = ln0;
	}
	auto* gamma = eg_make(g, 1, D, tag + ".gamma", gen, 0.0f, 1.0f, true);
	auto* scaled = eg_binary(g, S, D, ln0, gamma, CF::VECTOR_MUL_VECTOR);
	auto* beta = eg_make(g, 1, D, tag + ".beta", gen, 0.0f, 0.0f, true);
	return eg_binary(g, S, D, scaled, beta, CF::VECTOR_ADD_VECTOR);
}

// 搭 num_blocks 层 encoder，返回完整的计算图（前向顺序已保存在 owned 里）
static EncoderGraph build_encoder_graph(int S, int d_model, int heads, int d_ff,
										int num_blocks, bool residual, unsigned seed,
										float weight_scale = 0.3f)
{
	std::mt19937 gen(seed);
	EncoderGraph g;
	g.S = S;
	g.d_model = d_model;
	g.heads = heads;
	g.d_k = d_model / heads;
	g.d_ff = d_ff;
	g.blocks = num_blocks;
	g.residual = residual;

	g.x = eg_make(g, S, d_model, "x", gen, 1.0f, 0.0f, false);
	g.target = eg_make(g, S, d_model, "target", gen, 0.5f, 0.0f, false);

	const float sc = 1.0f / std::sqrt(static_cast<float>(g.d_k));

	// 正弦位置编码：PE[r][2k] = sin(r / 10000^(2k/d))，奇数列用 cos
	g.pos = eg_make(g, S, d_model, "pos", gen, 0.0f, 0.0f, false);
	for (int r = 0; r < S; ++r)
	{
		for (int k = 0; k < d_model / 2; ++k)
		{
			const float angle = static_cast<float>(r) /
				std::pow(10000.0f, 2.0f * k / static_cast<float>(d_model));
			g.pos->get_data()[r][2 * k] = std::sin(angle);
			g.pos->get_data()[r][2 * k + 1] = std::cos(angle);
		}
	}
	Tensor* cur = eg_binary(g, S, d_model, g.x, g.pos, CF::VECTOR_ADD_VECTOR);

	for (int b = 0; b < num_blocks; ++b)
	{
		const std::string pf = "b" + std::to_string(b) + ".";

		// ---- 多头自注意力（encoder：双向，无 causal mask）----
		Tensor* attn_sum = nullptr;
		for (int h = 0; h < heads; ++h)
		{
			const std::string hp = pf + "h" + std::to_string(h) + ".";
			auto* wq = eg_make(g, g.d_k, d_model, hp + "Wq", gen, weight_scale, 0.0f, true);
			auto* wk = eg_make(g, g.d_k, d_model, hp + "Wk", gen, weight_scale, 0.0f, true);
			auto* wv = eg_make(g, g.d_k, d_model, hp + "Wv", gen, weight_scale, 0.0f, true);
			auto* wo = eg_make(g, d_model, g.d_k, hp + "Wo", gen, weight_scale, 0.0f, true);

			// q/k/v = x @ W^T，复用「权重在左、按行输入在右」的 matvec
			auto* q = eg_binary(g, S, g.d_k, wq, cur, CF::MATRIX_MUL_VECTOR);
			auto* k = eg_binary(g, S, g.d_k, wk, cur, CF::MATRIX_MUL_VECTOR);
			auto* v = eg_binary(g, S, g.d_k, wv, cur, CF::MATRIX_MUL_VECTOR);

			auto* kt = eg_unary(g, g.d_k, S, k, CF::TRANSPOSE);
			auto* qk = eg_binary(g, S, S, q, kt, CF::MATRIX_MUL_MATRIX);
			auto* scale = eg_make(g, 1, S, hp + "scale", gen, 0.0f, sc, false);
			auto* scored = eg_binary(g, S, S, qk, scale, CF::VECTOR_MUL_VECTOR);
			auto* prob = eg_unary(g, S, S, scored, CF::SOFTMAX);
			if (g.attn_prob == nullptr)
			{
				g.attn_prob = prob;
			}

			auto* a = eg_binary(g, S, g.d_k, prob, v, CF::MATRIX_MUL_MATRIX);
			auto* y = eg_binary(g, S, d_model, wo, a, CF::MATRIX_MUL_VECTOR);
			attn_sum = (attn_sum == nullptr)
				? y
				: eg_binary(g, S, d_model, attn_sum, y, CF::VECTOR_ADD_VECTOR);
		}

		// ---- 残差 1 + LN ----
		auto* res1 = residual
			? eg_binary(g, S, d_model, cur, attn_sum, CF::VECTOR_ADD_VECTOR)
			: attn_sum;
		auto* ln1 = eg_layer_norm(g, res1, S, d_model, pf + "ln1", gen, g.ln_bare);
		if (g.ln1_out == nullptr)
		{
			g.ln1_out = ln1;
		}

		// ---- 逐位置 FFN：Linear -> GELU -> Linear ----
		auto* w1 = eg_make(g, d_ff, d_model, pf + "W1", gen, weight_scale, 0.0f, true);
		auto* b1 = eg_make(g, 1, d_ff, pf + "b1", gen, 0.0f, 0.0f, true);
		auto* w2 = eg_make(g, d_model, d_ff, pf + "W2", gen, weight_scale, 0.0f, true);
		auto* b2 = eg_make(g, 1, d_model, pf + "b2", gen, 0.0f, 0.0f, true);

		auto* z1 = eg_binary(g, S, d_ff, w1, ln1, CF::MATRIX_MUL_VECTOR);
		auto* z1b = eg_binary(g, S, d_ff, z1, b1, CF::VECTOR_ADD_VECTOR);
		auto* act = eg_unary(g, S, d_ff, z1b, CF::VECTOR_GELU);
		auto* z2 = eg_binary(g, S, d_model, w2, act, CF::MATRIX_MUL_VECTOR);
		auto* ffn = eg_binary(g, S, d_model, z2, b2, CF::VECTOR_ADD_VECTOR);

		// ---- 残差 2 + LN ----
		auto* res2 = residual
			? eg_binary(g, S, d_model, ln1, ffn, CF::VECTOR_ADD_VECTOR)
			: ffn;
		cur = eg_layer_norm(g, res2, S, d_model, pf + "ln2", gen, g.ln_bare);
	}

	g.output = cur;
	g.loss = eg_binary(g, 1, 1, cur, g.target, CF::VECTOR_SQUARED);

	g.watch = g.params;
	g.watch.push_back(g.x);
	return g;
}

static Tensor* eg_find_param(const EncoderGraph& g, const std::string& name)
{
	for (size_t i = 0; i < g.param_names.size(); ++i)
	{
		if (g.param_names[i] == name)
		{
			return g.params[i];
		}
	}
	return nullptr;
}

#endif // _ENCODER_BLOCK_H_
