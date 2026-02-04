LLM包含阶段：数据准备，模型设计，预训练(Pre-training)，微调(Fine-tuning)，评测与安全，部署与推理，持续迭代

LLM推理有两个阶段：Prefill和Decode。先通过 prefill 完成初始上下文的处理和缓存构建，再通过 decode 逐 token 生成内容。

## 01-大模型如何计算

当输入一段话后，模型做的事情有：
1. 把文字切分为token；
2. 把token经过embedding变成向量；
3. 对这些向量反复计算；
4. 算出“下一个最可能的token是什么”

大模型采用自回归生成方式，用已有的内容预测下一个token，无法一次性预测全部答案，也就决定了推理过程是“分步的”

## 02-Self-Attention自注意力机制

自注意力机制是深度学习中用于序列建模的核心机制，也是 Transformer 架构的基础组件。自注意力机制可以让模型在处理序列中某一元素时，动态关联序列内所有其他元素，依据关联强度分配权重，从而生成融合全局上下文的新特征表示。（全局上下文感知）

**核心**：替代 RNN、CNN 等传统序列建模方式，高效捕捉**长距离依赖**，支持并行计算，大幅提升训练与推理效率。

**标准计算流程**：

1. 给出输入序列嵌入矩阵 $X$ 
2. 生成 Q/K/V 向量：
	- 通过三组可学习权重矩阵 $W_Q,W_K,W_V$ ，将输入线性映射为查询向量 $Q$ ，键向量 $K$ ，值向量 $V$ 
3. 计算注意力原始分数
	- 用 $Q$ 与 $K^T$ 做点积，衡量元素之间的关联度，为避免高维向量导致数值过大，除以缩放因子  $\sqrt{d_k}$ （ $d_k$ 为 $K$ 的维度）：
	- $Score={QK^T\over\sqrt{d_k}}$
4. 归一化得到注意力权重
	- 通过 Softmax 函数将分数转为 0-1 之间的概率分布，确保权重和为1：
	- $Attention Weight=Softmax(score)$
5. 加权求和输出结果
	- 用归一化权重对 $V$ 加权求和，得到融合上下文的最终输出：
	- $Output=Attention Weight \cdot V$

整体公式整合：
$$Self\text{-}Attention(X)=Softmax({XW_Q \cdot (XW_K)^T \over \sqrt {d_K}}) \cdot XW_V$$

**补充设计：**
1. **多头自注意力**：将 Q/K/V 拆分为多个子空间并行计算自注意力，再拼接结果，可捕捉不同维度、不同类型的语义关联，提升模型表达能力。
2. **掩码自注意力**：在解码器阶段使用，通过掩码矩阵将未来位置的分数置为极小值，使模型生成序列时无法窥探后续未生成内容，保证生成逻辑合规。

## 03-Prefill 预填充
### 3.1-作用
prefill 是推理的第一个阶段，核心是一次性处理完整的提示词序列，计算所有位置的 Key、Value向量，并将这些缓存到 KV Cache中；同时输出第一个待生成的token
### 3.2-具体流程
假设 prompt 为“写一首关于春天的诗”，提示词被分词为【写，一首，关于，春天，的，诗】（6个token）
1. 模型将这6个token的嵌入向量作为输入，一次性计算所有位置的 Q、K、V 向量；
2. 执行自注意力计算：每个 token 的 Q 与所有 token K计算注意力权重，再与 V 加权求和，得到融合完整上下文的特征；
3. 通过输出层得到第一个生成 token 的概率分布（比如“春”）；
4. 将这6个 token 的 K/V 向量全部缓存到 KV Cache 中，供后续 decode 阶段复用
### 3.3-特点
1. prefill 处理的是整段序列，支持并行计算，效率高
2. 只会执行一次，prefill 是 decode 阶段的“准备工作”
3. 会占用一定的内存（缓存所有提示词的 K/V）
## 04-Decode 解码
### 4.1 作用
decode 是 prefill 之后的循环阶段。

逐一生成新的 token，每生成一个 token 时，仅计算该 token 的 Q 向量，复用 KV Cache 中已缓存的所有历史 K/V（提示词 + 已生成的 token），计算注意力并输出下一个 token；同时将新生成 token 的 K/V 追加到 KV Cache中。
### 4.2 具体流程
使用上面的例子，此时已生成第一个 token“春”，接下来生成第二个 token：
1. 取上一步生成的 token ”春“的嵌入向量作为输入，仅计算这个 token 的 Q、K、V向量
2. 执行自注意力计算：
	- Q：仅用“春”的 Q 向量
	- K：使用 KV cache 中 6 个提示词的 K 向量
	- 计算注意力权重，与 6 个提示词的 V 加权求和
3. 通过输出层得到第二个 token 的概率分布（假设为”风“）
4. 把“春”的 K/V 向量追加到 KV Cache 中
5. 重复步骤直到生成结束符或达到最大长度
### 4.3 特点
1. 处理单个的 token，只能串行计算
2. 每次计算仅需处理1个 token 的 Q，复用所有历史 K/V，计算量小（相比 prefill）
3. 每次生成后更新 KV Cache，缓存大小随生成 token 数量递增

## KV Cache
- KVCache 应用于推理阶段（也就是K、V的值是不变的）
- KVCache只存在于Decoder解码器中，它的目的是加速Q K V的两次矩阵相乘时的速度
- KVCache会加大内存占用
- KVCache 成立条件：简单概括，每一个 token 的输出只依赖于它自己以及之前的输入，与之后的输入无关。（在 transformer 模型中，BERT类 encoder 模型不满足这一性质，GPT 类 decoder 模型因使用了 causal mask 所以满足这一性质）

$$Attention(Q,K,V)=softmax({{QK^T}\over\sqrt{d_{key}}}+mask)V$$

> Q：目前的LLM基本都是 decoder-only 的结构，KVCache是否适用于所有的LLM呢？
>
> A：在输入预处理层中，通常会把 token ID 转换成 word embedding，然后加上 positional embedding。问题就出在 positional embedding 上：例如一些 ReRope之类的技术，在增加新的 token 时会把整个序列的 positional embedding 进行调整。 **对于同一个 token，上次的 token embedding 和这次的 token embedding 不相同，所以 KVCache 的条件不再成立。** 而一旦输入预处理层不满足 KVCache 的条件，后续 transformer 层的输入（即预处理层的输出）就发生了改变，不再适用 KVCache。

下图展示使用 KVCache 和不使用的对比
![kvcache](./picture/kvcache.jpg)
