LLM包含阶段：数据准备，模型设计，预训练(Pre-training)，微调(Fine-tuning)，评测与安全，部署与推理，持续迭代
LLM推理有两个阶段：Prefill和Decode
## Prefill
根据输入tokens生成第一个输出token，通过一次Forward就可以完成
在Forward中，输入Tokens间可以并行执行，因此执行效率很高

## KV Cache
- KV Cache 应用于推理阶段（也就是K、V的值是不变的）
- KV Cache只存在于Decoder解码器中，它的目的是加速Q K V的两次矩阵相乘时的速度
- KV Cache会加大内存占用
$$Attention(Q,K,V)=softmax({{QK^T}\over\sqrt{d_{key}}}+mask)V$$
> 为什么Q不用进行cache？
> 因为除去最新参与计算的一行，其余上面行数的计算结果已经cache了，不再需要也就不用cache

> ==KV Cache中缓存的有K矩阵、V矩阵，那么缓存的有$QK^T$矩阵吗？==

