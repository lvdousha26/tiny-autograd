// 激活函数的前向取值与梯度校验（中心差分）
#include <cmath>
#include <cstdio>
#include <vector>

#include "Graph.h"
#include "Tensor.h"

using CF = Tensor::CalculateFun;

struct Case
{
	const char* name;
	CF fn;
	float alpha;
	float probe;   // 用于打印前向值的输入
	float expect;  // 该输入下的解析前向值
};

static void run(const Case& c)
{
	const int cols = 5;
	Tensor x(1, cols, true);
	const float vals[5] = { -2.0f, -0.5f, 0.3f, 0.7f, 1.5f };
	for (int i = 0; i < cols; ++i)
	{
		x.get_data()[0][i] = vals[i];
	}

	Tensor act(1, cols, true);
	act.set_left(&x);
	act.set_calculate_fn(c.fn);
	act.set_alpha(c.alpha);

	Tensor target(1, cols);
	for (int i = 0; i < cols; ++i)
	{
		target.get_data()[0][i] = 0.3f;
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
	x.zero_grad();
	act.zero_grad();
	loss.zero_grad();
	loss.one_grad();
	graph::backward_memo(&loss, { &x });

	const float h = 1e-3f;
	double worst = 0.0;
	for (int i = 0; i < cols; ++i)
	{
		const float analytic = x.get_grad()[0][i];
		const float orig = x.get_data()[0][i];

		x.get_data()[0][i] = orig + h;
		forward_all();
		const float lp = loss.get_data()[0][0];
		x.get_data()[0][i] = orig - h;
		forward_all();
		const float lm = loss.get_data()[0][0];
		x.get_data()[0][i] = orig;

		const float numeric = (lp - lm) / (2.0f * h);
		const double denom = std::max(1e-3, std::fabs(static_cast<double>(numeric)));
		worst = std::max(worst, std::fabs(numeric - analytic) / denom);
	}
	forward_all();

	// 用第一个探测点展示前向值
	printf("%-12s f(%.2f)=%8.5f  expect %8.5f  max grad relerr %.2e\n",
		   c.name, static_cast<double>(c.probe), static_cast<double>(act.get_data()[0][0]),
		   static_cast<double>(c.expect), worst);
}

int main()
{
	printf("activation: forward value and gradient check (central diff, h=1e-3)\n\n");
	run({ "none",       CF::VECTOR_ASSIGN,     0.0f,  -2.0f, -2.0f });
	run({ "relu",       CF::VECTOR_RELU,       0.0f,  -2.0f,  0.0f });
	run({ "leaky_relu", CF::VECTOR_LEAKY_RELU, 0.01f, -2.0f, -0.02f });
	run({ "elu",        CF::VECTOR_ELU,        1.0f,  -2.0f, -0.86466f });
	run({ "sigmoid",    CF::VECTOR_SIGMOID,    0.0f,  -2.0f,  0.11920f });
	run({ "tanh",       CF::VECTOR_TANH,       0.0f,  -2.0f, -0.96403f });
	run({ "gelu",       CF::VECTOR_GELU,       0.0f,  -2.0f, -0.04550f });
	run({ "silu",       CF::VECTOR_SILU,       0.0f,  -2.0f, -0.23841f });
	return 0;
}
