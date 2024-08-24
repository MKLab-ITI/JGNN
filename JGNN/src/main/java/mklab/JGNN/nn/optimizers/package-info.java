/**
 * Contains optimizers that can be used to update training losses. Instantiate
 * optimizers, and let methods like
 * {@link mklab.JGNN.nn.Model#train(mklab.JGNN.nn.Loss, mklab.JGNN.nn.Optimizer, java.util.List, java.util.List)}
 * request parameter update rules given the internally computed outcome of
 * backprogagation. When writing training procedure of your own, use the
 * {@link mklab.JGNN.nn.optimizers.BatchOptimizer} to wrap some base optimizer
 * and accumulate gradient updates until calling
 * {@link mklab.JGNN.nn.optimizers.BatchOptimizer#updateAll()} at the end of
 * each batch or epoch.
 * 
 * @author Emmanouil Krasanakis
 */
package mklab.JGNN.nn.optimizers;