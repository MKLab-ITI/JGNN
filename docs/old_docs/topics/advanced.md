# Advanced practices

Several methods have been proposed as improvements to the basic message passing scheme. However, they tend to provide marginal accuracy improvements at the cost of increased computational complexity. For large graphs, it's best to avoid complex architectures since JGNN is designed to be lightweight and does not leverage GPU acceleration. Nevertheless, JGNN supports the following enhancements, which can be useful in scenarios where runtime is less critical (e.g., transfer learning, stream learning) or for analyzing smaller graphs:

- **Edge dropout**: Apply dropout to the adjacency matrix on each layer using `.layer("h{l+1}=dropout(A,0.5) @ h{l}")`. This operation disables certain caching optimizations under the hood.
  
- **Heterogeneity**: Some recent approaches account for high-pass frequency diffusion by including the graph Laplacian. This can be inserted into the architecture as a constant, for example: `.constant("L", adjacency.negative().cast(Matrix.class).setMainDiagonal(1))`.

- **Edge attention**: Computes new edge weights by taking the dot product of edge nodes using the formula `A.(h^T h)`, where `A` is a sparse adjacency matrix, the dot `.` represents the Hadamard product (element-wise multiplication), and `h` is a dense matrix containing node representations. JGNN efficiently implements this operation using the Neuralang function `att(A, h)`. For example, to create weighted adjacency matrices for each layer in gated attention networks: `.operation("A{l} = L1(nexp(att(A, h{l})))")`.

- **General message passing**: JGNN supports a fully generalized message-passing scheme for more complex relational analyses, such as those described by [Velickovic, 2022](https://arxiv.org/pdf/2202.11097.pdf). In this generalization, each edge transforms and propagates representations to node neighbors. You can create message matrices by gathering features from edge source and destination nodes. To obtain edge source indexes, use `src=from(A)`, and for destination indexes, use `dst=to(A)` where `A` is the adjacency matrix. Then use the horizontal concatenation operation `|` to combine node features. After constructing messages, any ad-hoc processing can be applied using traditional matrix operations. Make sure to define the correct matrix sizes for dense transformations, such as doubling the number of columns in `h{l}`. For any `LayeredBuilder`, ensure that `message{l}` is used to obtain a message from `h{l}` that is not shared with future layers. Receiver mechanisms usually perform some form of reduction on messages, which JGNN implements via summation. This reduction has the same expressive power as maximum-based reduction but is easier to backpropagate through. Perform this as follows:

```java
modelBuilder
    .operation("src=from(A)")
    .operation("dst=to(A)")
    .operation("message{l}=h{l}[src] | h{l}[dst]") // two times the number of h{l}'s features
    .operation("transformed_message{l}=...") // apply transformations
    .operation("received{l}=reduce(transformed_message{l}, A)");
```

