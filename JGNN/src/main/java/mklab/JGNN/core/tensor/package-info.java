/**
 * Contains implementations of tensor classes, as well as transparent access to
 * parts of these classes. Depending on the type of tensor, internal data can be
 * sparse or dense, with dense tensors being further subdivided in traditional
 * Java implementations and implementations levering SIMD (Single Input/Multiple
 * Data) optimizations.
 * 
 * @author Emmanouil Krasanakis
 */
package mklab.JGNN.core.tensor;