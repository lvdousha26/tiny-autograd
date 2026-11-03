#ifndef _GRAPH_H_
#define _GRAPH_H_
#include <vector>
#include "Tensor.h"

// 与具体网络层无关的计算图遍历算法
namespace graph
{
	// 自底向上：显式栈做拓扑逆序遍历
	// 前提：每个中间张量只有一个消费者（树形图），遇到 DAG 会重复累加
	void backward_tree(Tensor* loss);

	// 自顶向下：记忆化搜索（DP），树形图与 DAG 都正确
	// params 为需要梯度的参数张量，从它们向 loss 方向递归
	void backward_memo(Tensor* loss, const std::vector<Tensor*>& params);
}

#endif //_GRAPH_H_
