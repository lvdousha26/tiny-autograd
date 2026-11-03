// 单头自注意力演示：用计算图搭出 transformer 的核心单元，并做梯度校验
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "Graph.h"
#include "Tensor.h"

using CF = Tensor::CalculateFun;

// 图结构（S 个 token，模型维度 d）：
//   q = x * Wq^T, k = x * Wk^T, v = x * Wv^T
//   s = q * k^T / sqrt(d)      -> softmax 每行 -> p
//   a = p * v
//   y = a * Wo^T
//   loss = 0.5 * sum((y - target)^2) / S
struct Attention
{
	int S = 0;
	int d = 0;
	Tensor* x = nullptr;
	Tensor* target = nullptr;
	Tensor* wq = nullptr;
	Tensor* wk = nullptr;
	Tensor* wv = nullptr;
	Tensor* wo = nullptr;
	Tensor* loss = nullptr;
	std::vector<Tensor*> owned;
};

static Tensor* make_param(int rows, int cols, std::mt19937& gen)
{
	std::normal_distribution<float> dist(0.0f, 0.3f);
	auto* t = new Tensor(rows, cols, true);
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			t->get_data()[i][j] = dist(gen);
		}
	}
	return t;
}

static Attention build(int S, int d, unsigned seed)
{
	std::mt19937 gen(seed);
	std::normal_distribution<float> dist(0.0f, 1.0f);

	Attention m;
	m.S = S;
	m.d = d;

	m.x = new Tensor(S, d);
	for (int i = 0; i < S; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			m.x->get_data()[i][j] = dist(gen);
		}
	}
	m.target = new Tensor(S, d);
	for (int i = 0; i < S; ++i)
	{
		for (int j = 0; j < d; ++j)
		{
			m.target->get_data()[i][j] = 0.5f * dist(gen);
		}
	}

	m.wq = make_param(d, d, gen);
	m.wk = make_param(d, d, gen);
	m.wv = make_param(d, d, gen);
	m.wo = make_param(d, d, gen);

	// q/k/v = x * W^T，复用已有的 MATRIX_MUL_VECTOR（左为权重，右为按行的输入）
	auto* q = new Tensor(S, d, true);
	q->set_left(m.wq);
	q->set_right(m.x);
	q->set_calculate_fn(CF::MATRIX_MUL_VECTOR);

	auto* k = new Tensor(S, d, true);
	k->set_left(m.wk);
	k->set_right(m.x);
	k->set_calculate_fn(CF::MATRIX_MUL_VECTOR);

	auto* v = new Tensor(S, d, true);
	v->set_left(m.wv);
	v->set_right(m.x);
	v->set_calculate_fn(CF::MATRIX_MUL_VECTOR);

	// s = q * k^T：先转置 k，再做矩阵乘
	auto* kt = new Tensor(d, S, true);
	kt->set_left(k);
	kt->set_calculate_fn(CF::TRANSPOSE);

	auto* qk = new Tensor(S, S, true);
	qk->set_left(q);
	qk->set_right(kt);
	qk->set_calculate_fn(CF::MATRIX_MUL_MATRIX);

	// 缩放 1/sqrt(d)
	auto* scale = new Tensor(S, S);
	const float sc = 1.0f / std::sqrt(static_cast<float>(d));
	for (int i = 0; i < S; ++i)
	{
		for (int j = 0; j < S; ++j)
		{
			scale->get_data()[i][j] = sc;
		}
	}
	auto* scored = new Tensor(S, S, true);
	scored->set_left(qk);
	scored->set_right(scale);
	scored->set_calculate_fn(CF::VECTOR_MUL_VECTOR);

	// 每行 softmax
	auto* p = new Tensor(S, S, true);
	p->set_left(scored);
	p->set_calculate_fn(CF::SOFTMAX);

	// a = p * v
	auto* a = new Tensor(S, d, true);
	a->set_left(p);
	a->set_right(v);
	a->set_calculate_fn(CF::MATRIX_MUL_MATRIX);

	// y = a * Wo^T
	auto* y = new Tensor(S, d, true);
	y->set_left(m.wo);
	y->set_right(a);
	y->set_calculate_fn(CF::MATRIX_MUL_VECTOR);

	m.loss = new Tensor(1, 1, true);
	m.loss->set_left(y);
	m.loss->set_right(m.target);
	m.loss->set_calculate_fn(CF::VECTOR_SQUARED);

	m.owned = { m.x, m.target, m.wq, m.wk, m.wv, m.wo, q, k, v, kt, qk, scale, scored, p, a, y, m.loss };
	return m;
}

static void forward_all(Attention& m)
{
	for (auto* t : m.owned)
	{
		t->forward_calculate_data();
	}
}

static void zero_all(Attention& m)
{
	for (auto* t : m.owned)
	{
		t->zero_grad();
	}
}

// 中心差分校验：解析梯度来自反向传播，数值梯度来自扰动后重新前向
static void grad_check(Attention& m, Tensor* param, const char* name)
{
	std::vector<Tensor*> params = { m.wq, m.wk, m.wv, m.wo };
	forward_all(m);
	zero_all(m);
	m.loss->one_grad();
	graph::backward_memo(m.loss, params);

	const float h = 1e-2f;
	double worst = 0.0;

	for (int i = 0; i < param->get_row_dimension(); ++i)
	{
		for (int j = 0; j < param->get_col_dimension(); ++j)
		{
			if ((i * 7 + j * 3) % 11 != 0)
			{
				continue;
			}
			const float analytic = param->get_grad()[i][j];
			const float orig = param->get_data()[i][j];

			param->get_data()[i][j] = orig + h;
			forward_all(m);
			const float lp = m.loss->get_data()[0][0];

			param->get_data()[i][j] = orig - h;
			forward_all(m);
			const float lm = m.loss->get_data()[0][0];

			param->get_data()[i][j] = orig;
			const float numeric = (lp - lm) / (2.0f * h);

			const double denom = std::max(1e-4, std::fabs(static_cast<double>(numeric)));
			const double rel = std::fabs(numeric - analytic) / denom;
			if (rel > worst)
			{
				worst = rel;
			}
		}
	}
	forward_all(m);
	printf("%-6s max relative error = %.3e\n", name, worst);
}

int main()
{
	const int S = 6;
	const int d = 8;
	Attention m = build(S, d, 2026u);

	forward_all(m);
	printf("--- forward ---\n");
	printf("loss = %.6f\n", static_cast<double>(m.loss->get_data()[0][0]));

	// 打印注意力矩阵，确认每行和为 1
	printf("attention (each row sums to 1):\n");
	const Tensor* p = m.owned[13];
	for (int i = 0; i < S; ++i)
	{
		float sum = 0.0f;
		printf("  ");
		for (int j = 0; j < S; ++j)
		{
			printf("%.3f ", static_cast<double>(p->get_data()[i][j]));
			sum += p->get_data()[i][j];
		}
		printf(" = %.3f\n", static_cast<double>(sum));
	}

	printf("--- gradient check (central difference) ---\n");
	grad_check(m, m.wq, "Wq");
	grad_check(m, m.wk, "Wk");
	grad_check(m, m.wv, "Wv");
	grad_check(m, m.wo, "Wo");

	printf("--- train on a fixed target ---\n");
	const float lr = 0.05f;
	std::vector<Tensor*> params = { m.wq, m.wk, m.wv, m.wo };
	for (int step = 0; step <= 300; ++step)
	{
		forward_all(m);
		const float loss = m.loss->get_data()[0][0];
		if (step % 50 == 0)
		{
			printf("step %3d  loss = %.6f\n", step, static_cast<double>(loss));
		}
		zero_all(m);
		m.loss->one_grad();
		graph::backward_memo(m.loss, params);
		for (auto* t : params)
		{
			for (int i = 0; i < t->get_row_dimension(); ++i)
			{
				for (int j = 0; j < t->get_col_dimension(); ++j)
				{
					t->get_data()[i][j] -= lr * t->get_grad()[i][j];
				}
			}
		}
	}
	return 0;
}
