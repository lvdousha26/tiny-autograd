// 图结构与动态建图的实验程序
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "Graph.h"
#include "Tensor.h"

using CalculateFun = Tensor::CalculateFun;

// 带共享子表达式的 DAG：
//   u = x + c1
//   p = u + c2
//   q = u - c2      <- u 有两个消费者，所以是 DAG 而不是树
//   s = p + q
//   L = 0.5 * (s - t)^2
// 解析解：s = 2*x + 2*c1，ds/dx = 2，dL/dx = 2*(s - t)
struct DagGraph
{
	Tensor* x = nullptr;
	Tensor* loss = nullptr;
	std::vector<Tensor*> owned;
};

static DagGraph build_dag(float x_val, float c1_val, float c2_val, float t_val)
{
	DagGraph g;
	g.x = new Tensor(1, 1, true);
	auto* c1 = new Tensor(1, 1);
	auto* c2 = new Tensor(1, 1);
	auto* t = new Tensor(1, 1);
	auto* u = new Tensor(1, 1, true);
	auto* p = new Tensor(1, 1, true);
	auto* q = new Tensor(1, 1, true);
	auto* s = new Tensor(1, 1, true);
	g.loss = new Tensor(1, 1, true);

	g.x->get_data()[0][0] = x_val;
	c1->get_data()[0][0] = c1_val;
	c2->get_data()[0][0] = c2_val;
	t->get_data()[0][0] = t_val;

	u->set_left(g.x);
	u->set_right(c1);
	u->set_calculate_fn(CalculateFun::VECTOR_ADD_VECTOR);
	p->set_left(u);
	p->set_right(c2);
	p->set_calculate_fn(CalculateFun::VECTOR_ADD_VECTOR);
	q->set_left(u);
	q->set_right(c2);
	q->set_calculate_fn(CalculateFun::VECTOR_SUB_VECTOR);
	s->set_left(p);
	s->set_right(q);
	s->set_calculate_fn(CalculateFun::VECTOR_ADD_VECTOR);
	g.loss->set_left(s);
	g.loss->set_right(t);
	g.loss->set_calculate_fn(CalculateFun::VECTOR_SQUARED);

	g.owned = { g.x, c1, c2, t, u, p, q, s, g.loss };
	for (auto* node : g.owned)
	{
		node->forward_calculate_data(); // 按拓扑序前向，叶子节点是空操作
	}
	return g;
}

static void free_dag(DagGraph& g)
{
	for (auto* node : g.owned)
	{
		delete node;
	}
	g.owned.clear();
}

static void test_dag()
{
	const float x_val = 1.0f, c1 = 2.0f, c2 = 3.0f, t = 10.0f;
	const float s_val = 2.0f * x_val + 2.0f * c1;
	const float loss_val = 0.5f * (s_val - t) * (s_val - t);
	const float dx_val = 2.0f * (s_val - t);

	printf("\n--- PART 1: DAG, node u has two consumers ---\n");
	printf("analytic      : loss=%.4f  dL/dx=%.4f\n", loss_val, dx_val);

	DagGraph g1 = build_dag(x_val, c1, c2, t);
	graph::backward_tree(g1.loss);
	printf("stack dfs     : loss=%.4f  dL/dx=%.4f\n", g1.loss->get_data()[0][0], g1.x->get_grad()[0][0]);
	free_dag(g1);

	DagGraph g2 = build_dag(x_val, c1, c2, t);
	graph::backward_memo(g2.loss, { g2.x });
	printf("memo search   : loss=%.4f  dL/dx=%.4f\n", g2.loss->get_data()[0][0], g2.x->get_grad()[0][0]);
	free_dag(g2);
}

// 动态建图：图的形状由运行时数据决定，静态图无法表达
static void test_dynamic(float x_val)
{
	Tensor* x = new Tensor(1, 1, true);
	x->get_data()[0][0] = x_val;
	std::vector<Tensor*> owned{ x };
	Tensor* cur = x;

	const int depth = 1 + static_cast<int>(std::fabs(x_val));
	for (int i = 0; i < depth; ++i)
	{
		Tensor* node = new Tensor(1, 1, true);
		node->set_left(cur);
		node->set_calculate_fn(CalculateFun::VECTOR_RELU);
		node->forward_calculate_data();
		owned.push_back(node);
		cur = node;
	}

	Tensor* target = new Tensor(1, 1);
	target->get_data()[0][0] = 0.0f;
	Tensor* loss = new Tensor(1, 1, true);
	loss->set_left(cur);
	loss->set_right(target);
	loss->set_calculate_fn(CalculateFun::VECTOR_SQUARED);
	loss->forward_calculate_data();
	owned.push_back(target);
	owned.push_back(loss);

	graph::backward_memo(loss, { x });
	printf("x=%5.2f  nodes=%2d  relu_depth=%d  loss=%.4f  dL/dx=%.4f\n",
		   static_cast<double>(x_val), static_cast<int>(owned.size()), depth,
		   static_cast<double>(loss->get_data()[0][0]), static_cast<double>(x->get_grad()[0][0]));

	for (auto* node : owned)
	{
		delete node;
	}
}

static void test_rebuild_cost(int iters)
{
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; ++i)
	{
		DagGraph g = build_dag(1.0f, 2.0f, 3.0f, 10.0f);
		graph::backward_memo(g.loss, { g.x });
		free_dag(g);
	}
	auto t1 = std::chrono::steady_clock::now();

	DagGraph g = build_dag(1.0f, 2.0f, 3.0f, 10.0f);
	auto t2 = std::chrono::steady_clock::now();
	for (int i = 0; i < iters; ++i)
	{
		for (auto* node : g.owned)
		{
			node->zero_grad();
		}
		for (auto* node : g.owned)
		{
			node->forward_calculate_data();
		}
		graph::backward_memo(g.loss, { g.x });
	}
	auto t3 = std::chrono::steady_clock::now();
	free_dag(g);

	const double rebuilt = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
	const double reused = std::chrono::duration<double, std::micro>(t3 - t2).count() / iters;
	printf("\n--- PART 3: rebuild vs reuse, %d iters ---\n", iters);
	printf("rebuild each time : %.3f us/iter\n", rebuilt);
	printf("build once, reuse : %.3f us/iter  (%.1fx faster)\n", reused, rebuilt / reused);
}

int main()
{
	test_dag();

	printf("\n--- PART 2: dynamic graph, shape depends on data ---\n");
	test_dynamic(0.5f);
	test_dynamic(1.7f);
	test_dynamic(3.2f);

	test_rebuild_cost(20000);
	return 0;
}
