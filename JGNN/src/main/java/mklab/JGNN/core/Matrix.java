package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.matrix.AccessRow;
import mklab.JGNN.core.matrix.AccessCol;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.TransposedMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.SparseTensor;

import java.util.Map.Entry;

/**
 * This class provides an abstract implementation of Matrix functionalities.
 * Matrices inherit {@link Tensor} operations, such as addition, 
 * element-by-element multiplication, randomizing them and producing zero copies.
 * Additionally, matrix multiplication, transposition and access operations are
 * provided.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Matrix extends Tensor {
	private long rows;
	private long cols;
	
	public Matrix(long rows, long cols) {
		super(rows*cols);
		this.rows = rows;
		this.cols = cols;
	}
	
	public abstract Iterable<Entry<Long, Long>> getNonZeroEntries();
	
	protected Matrix() {
	}
	
	@Override
	public final Matrix zeroCopy() {
		return zeroCopy(rows, cols);
	}
	
	public abstract Matrix zeroCopy(long rows, long cols);
	
	public final long getRows() {
		return rows;
	}
	
	public final long getCols() {
		return cols;
	}
	
	public final double get(long row, long col) {
		if(row<0 || col<0 || row>=rows || col>=cols)
			throw new IllegalArgumentException("Element out of range ("+row+","+col+") for "+describe());
		return get(row+col*rows);
	}
	
	public final Matrix put(long row, long col, double value) {
		put(row+col*rows, value);
		return this;
	}
	/**
	 * Creates a transposed copy of the matrix.
	 * Note: Contrary to typical tensor operations, in-place transposition is not supported.
	 * However, related methods can help avoid explicit transposition without allocating more
	 * memory.
	 * @return A transposed copy of the matrix.
	 * @see #matmul(Matrix)
	 * @see #asTransposed()
	 */
	public final Matrix transposed() {
		Matrix ret = zeroCopy(getCols(), getRows());
		for(Entry<Long, Long> element : getNonZeroEntries())
			ret.put(element.getValue(), element.getKey(), get(element.getKey(), element.getValue()));
		return ret;
	}
	/**
	 * Creates a transposed version of the matrix that accesses the same elements (thus, editing one
	 * edits the other) without allocating additional memory.
	 * @return A {@link TransposedMatrix}.
	 */
	public Matrix asTransposed() {
		return new TransposedMatrix(this);
	}
	/**
	 * Performs the linear algebra transformation A*x where A is this matrix and x a vector
	 * @param x The one-dimensional tensor which is the vector being transformed.
	 * @return The one-dimensional outcome of the transformation.
	 */
	public final Tensor transform(Tensor x) {
		x.assertSize(cols);
		Tensor ret = new DenseTensor(rows);
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			ret.put(row, ret.get(row) + get(row, col)*x.get(col));
		}
		return ret;
	}
	/**
	 * Performs the matrix multiplication of <code>this*with</code> and the recipient.
	 * 
	 * @param with The matrix to multiply with.
	 * @return A matrix that stores the outcome of the multiplication.
	 * @see #matmul(Matrix)
	 */
	public final Matrix matmul(Matrix with) {
		if(cols!=with.getRows())
			throw new IllegalArgumentException("Mismatched matrix sizes between "+describe()+" and "+with.describe());
		Matrix ret = with.zeroCopy(getRows(), with.getCols());
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			for(long col2=0;col2<with.getCols();col2++) 
				ret.put(row, col2, ret.get(row, col2) + get(row, col)*with.get(col, col2));
		}
		return ret;
	}

	/**
	 * Can be used to perform fast computation of the matrix multiplications <code>this*with</code>,
	 * <code>this.transposed()*with</code>, <code>this*with.transposed()</code>, 
	 * <code>this.transposed()*with.transposed</code> while avoiding the overhead of calling
	 * {@link #transposed()}. In this first of those cases, this operation
	 * becomes equivalent to {@link #matmul(Matrix)}.
	 * 
	 * @param with The matrix to multiply with.
	 * @param transposeSelf Whether <code>this</code> matrix should be transposed before multiplication.
	 * @param transposeWith Whether the multiplied <code>with</code> matrix should be transposed before multiplication.
	 * @return A matrix that stores the outcome of the multiplication.
	 * @see #matmul(Matrix)
	 * @see #transposed()
	 */
	public final Matrix matmul(Matrix with, boolean transposeSelf, boolean transposeWith) {
		if((transposeSelf?rows:cols)!=(transposeWith?with.getCols():with.getRows()))
			throw new IllegalArgumentException("Mismatched matrix sizes");
		Matrix ret = with.zeroCopy(transposeSelf?cols:rows, transposeWith?with.getRows():with.getCols());
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = transposeSelf?element.getValue():element.getKey();
			long col = transposeSelf?element.getKey():element.getValue();
			for(long col2=0;col2<(transposeWith?with.getRows():with.getCols());col2++) 
				ret.put(row, col2, ret.get(row, col2) + get(element.getKey(),element.getValue())*with.get(transposeWith?col2:col, transposeWith?col:col2));
		}
		return ret;
	}
	
	public static Matrix external(Tensor horizontal, Tensor vertical) {
		Matrix ret = new DenseMatrix(horizontal.size(), vertical.size());
		for(long row=0;row<horizontal.size();row++)
			for(long col=0;col<vertical.size();col++) 
				ret.put(row, col, horizontal.get(row)*vertical.get(col));
		return ret;
	}
	
	@Override
	public Tensor selfMultiply(Tensor other) {
		if(other.size()==getCols()) {
			for(Entry<Long, Long> element : getNonZeroEntries()) {
				long row = element.getKey();
				long col = element.getValue();
				put(row, col, get(row, col)*other.get(col));
			}
			return this;
		}
		else
			return super.selfMultiply(other);
	}
	
	@Override
	protected boolean isMatching(Tensor other) {
		if(!(other instanceof Matrix)) {
			if(rows!=1 && cols!=1)
				return false;
			else
				return super.isMatching(other);
		}
		else if(rows!=((Matrix)other).rows || cols!=((Matrix)other).cols)
			return false;
		return true;
	}
	
	/*@Override
	public String toString() {
		String ret = "";
		for(long row=0;row<rows;row++) {
			if(cols>0)
				ret += get(row, 0);
			for(long col=1;col<cols;col++) 
				ret += ","+get(row, col);
			ret += "\n";
		}
		return "[\n"+ret+"]";
	}*/
	
	@Override
	public String describe() {
		return getClass().getSimpleName()+" ("+rows+","+cols+")";
	}
	
	public Matrix onesMask() {
		Matrix ones = zeroCopy(getRows(), getCols());
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			ones.put(row, col, 1.);
		}
		return ones;
	}
	
	public Matrix laplacian() {
		return ((Matrix)copy()).setToLaplacian();
	}
	
	public Matrix setToLaplacian() {
		HashMap<Long, Double> outDegrees = new HashMap<Long, Double>();
		HashMap<Long, Double> inDegrees = new HashMap<Long, Double>();
		for(Entry<Long,Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			double value = get(row, col);
			outDegrees.put(row, outDegrees.getOrDefault(row, 0.)+value);
			inDegrees.put(col, inDegrees.getOrDefault(col, 0.)+value);
		}
		for(Entry<Long,Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			double div = Math.sqrt(outDegrees.get(row)*inDegrees.get(col));
			if(div!=0)
				put(row, col, get(row, col)/div);
		}
		return this;
	}
	
	/**
	 * Retrieves the given row as a tensor. Editing the result
	 * also edits the original matrix.
	 * No new memory is allocated for matrix values.
	 * @param row The given row.
	 * @return A {@link AccessRow} instance of the corresponding row.
	 */
	public Tensor getRow(long row) {
		return new AccessRow(this, row);
	}

	/**
	 * Retrieves the given column as a tensor. Editing the result
	 * also edits the original matrix.
	 * No new memory is allocated for matrix values.
	 * @param col The given column.
	 * @return A {@link AccessCol} of the corresponding row.
	 */
	public Tensor getCol(long col) {
		return new AccessCol(this, col);
	}
	
	@Override
	public String toString() {
		StringBuilder res = new StringBuilder();
		for(int row=0;row<rows;row++) {
			res.append("[");
			for(int col=0;col<cols;col++) {
				if(col!=0)
					res.append(",");
				res.append(get(row, col));
			}
			res.append("]");
		}
		return res.toString();
	}
	
	public final static Matrix fromSparseColumns(List<Tensor> tensors) {
		Matrix ret = new SparseMatrix(tensors.get(0).size(), tensors.size());
		for(int col=0;col<tensors.size();col++) 
			for(long row : tensors.get(col).getNonZeroElements())
				ret.put(row, col, tensors.get(col).get(row));
		return ret;
	}
	
	public final List<Tensor> toSparseColumns() {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long col=0;col<getCols();col++)
			ret.add(new SparseTensor(getRows()));
		for(Entry<Long, Long> entry : getNonZeroEntries()) {
			long row = entry.getKey();
			long col = entry.getValue();
			ret.get((int)col).put(row, get(row, col));
		}
		return ret;
	}
	
	/**
	 * Converts a given value to a JGNN-compatible 1x1 matrix.
	 * @param value A given value.
	 * @return a Matrix holding the given value
	 * @see Tensor#fromDouble(double)
	 * @see Tensor#toDouble()
	 */
	public static Matrix fromDouble(double value) {
		Matrix ret = new DenseMatrix(1, 1);
		ret.put(0, 0, value);
		return ret;
	}
}
