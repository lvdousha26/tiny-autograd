//

//

#include <iostream>
#include "Graph.h"
#include "Net.h"

Net::Net() = default;

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

void Net::add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun activation_fn)
{
	if (layers.empty())
	{
		// 第一层
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn));
	}
	else
	{
		// 后续层：将前一层的输出作为当前层的输入
		layers.emplace_back(new FCLayer(num_of_inputs, num_of_outputs, activation_fn));

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
	y_label = new Tensor(1, last_layer_outputs->get_col_dimension());
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




