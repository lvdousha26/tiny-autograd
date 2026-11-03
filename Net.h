//

//

#ifndef _NET_H_
#define _NET_H_
#include <vector>
#include "Tensor.h"
#include "FCLayer.h"

class Net
{
 public:
	enum LossFun{MEAN_SQUARED, CROSS_ENTROPY}; // 损失函数类型

	// 反向传播的两种实现：显式栈（自底向上填表）与记忆化搜索（自顶向下）
	enum BackwardMode{STACK_DFS, MEMO_SEARCH};
	// batch 为一次前向处理的样本数，必须在 add_layer 之前确定
	explicit Net(int batch = 1);

	Net(const Net&) = delete;
	Net& operator=(const Net&) = delete;

	// 添加网络层
	void add_layer(int num_of_inputs, int num_of_outputs, FCLayer::ActivationFun fn);

	// 设置损失函数
	void set_loss_function(LossFun fn);

	// 训练和预测
	float train(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const;

	// 批量训练：一次前向反向处理整批样本，梯度已按批大小取均值
	float train_batch(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const;

	// 批量前向，返回整批的平均损失
	float forward_batch(const std::vector<std::vector<float>>& input_value, const std::vector<std::vector<float>>& label_value) const;

	// 前向传播
	float forward(const std::vector<float>& input_value, const std::vector<float>& label_value) const;

	// 反向传播
	void backward() const;

	// 记忆化搜索版反向传播（自顶向下 DP，需要 consumer 反向边）
	void backward_memo() const;

	// 切换反向传播实现
	void set_backward_mode(BackwardMode mode);

	void zero_grad() const;

	// 参数更新
	void SGD(float learn_rate) const;

	void init_parameters(float mean, float variance) const;
	void printParameters() const;
	[[nodiscard]] float get_loss_value() const;
	[[nodiscard]] const std::vector<float>& get_y_hat_softmax() const;
	[[nodiscard]] const std::vector<float>& get_y_hat_softmax(int row) const;
	[[nodiscard]] inline const std::vector<float>& get_y_hat() const;
	~Net();
 private:
	std::vector<FCLayer*> layers; 				// 网络层集合
	Tensor* y_label{};							// 标签张量
	Tensor* loss{};
	int batch_rows = 1;								// 损失张量
	void zero_grad_without_parameters() const;
	BackwardMode backward_mode = STACK_DFS;
};

inline float Net::get_loss_value() const
{
	return loss->get_data()[0][0];
}

inline const std::vector<float>& Net::get_y_hat_softmax() const
{
	return (*layers.rbegin())->GetOutputs()->get_softmax_data();
}

inline const std::vector<float>& Net::get_y_hat_softmax(int row) const
{
	return (*layers.rbegin())->GetOutputs()->get_softmax_batch()[row];
}

inline const std::vector<float>& Net::get_y_hat() const
{
	return (*layers.rbegin())->GetOutputs()->get_data()[0];
}

#endif //_NET_H_
