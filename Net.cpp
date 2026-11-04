//

//

#include <iostream>
#include "Graph.h"
#include "Net.h"

Net::Net(int batch) : batch_rows(batch) {}

Net::~Net()
{
	delete loss;
	delete y_label;
	for (const auto& it : layers)
	{
		delete it;
	}
}

void Net::SGD(float learn_rate) const
{
	for (const auto& it : layers)
	{
		it->update_parameters(learn_rate);
	}
}

void Net::init_parameters(float mean, float variance) const
{
	for (const auto& it : layers)
	{
		it->init_parameters(mean, variance);
	}
}

void Net::printParameters() const
{
	int l_no = 1;
	for (const auto& it : layers)
	{
		std::cout << "layer:" << l_no << ", ";
		it->printParameters();
		std::cout << std::endl;
		++l_no;
	}
}

float Net::train(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const
{
	float loss_value = 0.0f;
	for (int i = 0; i < input_value.size(); ++i)
	{
		forward(input_value[i], label_value[i]);
		loss_value += get_loss_value();
		zero_grad_without_parameters();
		if (backward_mode == MEMO_SEARCH)
		{
			backward_memo();
		}
		else
		{
			backward();
		}
	}
	return loss_value;
}

float Net::forward(const std::vector<float>& input_value, const std::vector<float>& label_value) const
{
	layers[0]->GetInputs()->set_data(input_value);
	y_label->set_data(label_value);
	for (const auto& it : layers)
	{
		it->calculate();
	}
	loss->forward_calculate_data();
	return get_loss_value();
}

float Net::forward_batch(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const
{
	layers[0]->GetInputs()->set_batch_data(input_value);
	y_label->set_batch_data(label_value);
	for (const auto& it : layers)
	{
		it->calculate();
	}
	loss->forward_calculate_data();
	return get_loss_value();
}

float Net::train_batch(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const
{
	const float loss_value = forward_batch(input_value, label_value);
	zero_grad_without_parameters();
	float sparse_term = 0.0f;
	if (sparse_coeff > 0.0f && sparse_layer >= 0 && sparse_layer < static_cast<int>(layers.size()))
	{
		// 直接把 lambda * mean(sum z) 的梯度注入隐层输出：
		// 反向传播用的是 +=，所以这里预置的梯度不会被后续 consumer 覆盖。
		Tensor* z = layers[sparse_layer]->GetOutputs();
		std::vector<std::vector<float>>& gz = z->get_grad();
		const std::vector<std::vector<float>>& zd = z->get_data();
		const float scale = sparse_coeff / static_cast<float>(batch_rows);
		for (int r = 0; r < z->get_row_dimension(); ++r)
		{
			for (int j = 0; j < z->get_col_dimension(); ++j)
			{
				if (zd[r][j] > 0.0f)
				{
					gz[r][j] += scale;
					sparse_term += zd[r][j];
				}
			}
		}
		sparse_term *= scale;
	}
	if (backward_mode == MEMO_SEARCH)
	{
		backward_memo();
	}
	else
	{
		backward();
	}
	return loss_value + sparse_term;
}

void Net::backward() const
{
	graph::backward_tree(loss);
}

void Net::set_backward_mode(BackwardMode mode)
{
	backward_mode = mode;
}

void Net::backward_memo() const
{
	std::vector<Tensor*> params;
	params.reserve(layers.size() * 2);
	for (const auto& it : layers)
	{
		params.push_back(it->GetWeights());
		params.push_back(it->GetBias());
	}
	graph::backward_memo(loss, params);
}

void Net::zero_grad() const
{
	for (const auto& it : layers)
	{
		it->zero_grad();
	}
}

void Net::zero_grad_without_parameters() const
{
	for (const auto& it : layers)
	{
		it->zero_grad_without_parameters();
	}
}

void Net::set_sparse_penalty(float coeff, int layer)
{
	sparse_coeff = coeff;
	sparse_layer = layer;
}

void Net::add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun activation_fn, int top_k)
{
	if (layers.empty())
	{
		// 第一层
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn, batch_rows, top_k));
	}
	else
	{
		// 后续层：将前一层的输出作为当前层的输入
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn, batch_rows, top_k));

		Tensor* cur_layer_inputs = (*layers.rbegin())->GetInputs();
		Tensor* pre_layer_outputs = (*(layers.rbegin() + 1))->GetOutputs();
		cur_layer_inputs->set_left(pre_layer_outputs);
		cur_layer_inputs->set_calculate_fn(Tensor::CalculateFun::VECTOR_ASSIGN);
	}
}

void Net::set_loss_function(LossFun fn)
{
	delete loss;
	delete y_label;
	loss = new Tensor(1, 1, true);
	Tensor* last_layer_outputs = (*layers.rbegin())->GetOutputs();
	y_label = new Tensor(batch_rows, last_layer_outputs->get_col_dimension());
	loss->set_left(last_layer_outputs);
	loss->set_right(y_label);
	switch (fn)
	{
	case CROSS_ENTROPY:
		loss->set_calculate_fn(Tensor::CalculateFun::VECTOR_CROSSENTROPY);
		break;
	case MEAN_SQUARED:
		loss->set_calculate_fn(Tensor::CalculateFun::VECTOR_SQUARED);
		break;
	}
}




