/**
 * Contains empty extensions of datatypes that hold only dimension names and
 * sizes but no ddata. These types are pervasive in that any operation they are
 * involved in has an empty outcome too. Main usage of empty data types is to
 * verify created model integrity in terms of operations without actually
 * performing any computations. For example, empty inputs are preferred for
 * {@link mklab.JGNN.adhoc.ModelBuilder#autosize(mklab.JGNN.core.Tensor...)}
 * 
 * @author Emmanouil Krasanakis
 */
package mklab.JGNN.core.empy;