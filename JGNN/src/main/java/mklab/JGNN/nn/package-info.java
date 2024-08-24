/**
 * Implements neural networks components that are combined to define GNNs or
 * other types of machine learning models. Hand-wiring everything may be
 * cumbersome, so prefer using {@link mklab.JGNN.adhoc.ModelBuilder} and its
 * extensions to construct {@link mklab.JGNN.nn.Model} instances. Components
 * matching common neural operations are provided in sub-packages, where they
 * are separated by their functional role as activations, inputs, operations, or
 * pooling functions. In addition to operations. Additionally, Java code
 * components are provided for losses and model parameter initialization.
 * 
 * @author Emmanouil Krasanakis
 */
package mklab.JGNN.nn;