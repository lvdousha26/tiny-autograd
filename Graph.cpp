#include <functional>
#include <stack>
#include <unordered_set>
#include "Graph.h"

void graph::backward_tree(Tensor* loss)
{
	if (!loss->is_require_grad() || loss->get_calculate_fn() == Tensor::LEAF)
	{
		return;
	}

	std::stack<Tensor*> tensor_stack;
	loss->one_grad();
	tensor_stack.push(loss);
	while (true)
	{
		if (tensor_stack.empty())
		{
			break;
		}
		Tensor* cur_tensor = tensor_stack.top();
		tensor_stack.pop();
		Tensor* left = cur_tensor->get_left();
		Tensor* right = cur_tensor->get_right();
		if (left != nullptr && left->is_require_grad())
		{
			cur_tensor->backward_calculate_left_grad();
			tensor_stack.push(left);
		}
		if (right != nullptr && right->is_require_grad())
		{
			cur_tensor->backward_calculate_right_grad();
			tensor_stack.push(right);
		}
	}
}

void graph::backward_memo(Tensor* loss, const std::vector<Tensor*>& params)
{
	if (!loss->is_require_grad() || loss->get_calculate_fn() == Tensor::LEAF)
	{
		return;
	}

	loss->one_grad();

	std::unordered_set<Tensor*> solved;
	std::function<void(Tensor*)> grad_of = [&](Tensor* z)
	{
		if (!z->is_require_grad() || !solved.insert(z).second)
		{
			return;
		}
		for (const auto& edge : z->get_consumers())
		{
			Tensor* consumer = edge.node;
			grad_of(consumer); // 先保证消费者的梯度已经完整
			if (edge.slot == 0)
			{
				consumer->backward_calculate_left_grad();
			}
			else
			{
				consumer->backward_calculate_right_grad();
			}
		}
	};

	for (auto* it : params)
	{
		grad_of(it);
	}
}
