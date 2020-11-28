package mklab.JGNN.core;

import java.util.Iterator;

import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range;

import java.util.Map.Entry;

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
	
	public final Matrix transposed() {
		Matrix ret = zeroCopy(getCols(), getRows());
		for(Entry<Long, Long> element : getNonZeroEntries())
			ret.put(element.getValue(), element.getKey(), get(element.getKey(), element.getValue()));
		return ret;
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
	
	public final Matrix matmul(Matrix with) {
		if(cols!=with.getRows())
			throw new IllegalArgumentException("Mismatched matrix sizes");
		Matrix ret = with.zeroCopy(getRows(), with.getCols());
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			for(long col2=0;col2<with.getCols();col2++) 
				ret.put(row, col2, ret.get(row, col2) + get(row, col)*with.get(col, col2));
		}
		return ret;
	}
	

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
	
	public static class MatrixCol extends Tensor {
		private Matrix matrix;
		private long col;
		public MatrixCol(Matrix matrix, long col) {
			super(matrix.getRows());
			this.matrix = matrix;
			this.col = col;
		}
		@Override
		protected void allocate(long size) {
		}
		@Override
		public Tensor put(long pos, double value) {
			matrix.put(pos, col, value);
			return this;
		}
		@Override
		public double get(long pos) {
			return matrix.get(pos, col);
		}
		@Override
		public Tensor zeroCopy() {
			throw new RuntimeException("MatrixCol is used to access matrix elements only (consider using it as a second argument)");
		}
		@Override
		public Iterator<Long> traverseNonZeroElements() {
			return new Range(0, size());
		}
		
	}
	
	/*public static class Vector extends Matrix {
		private Tensor tensor;
		public Vector(Tensor tensor) {
			super(tensor.size(), 1);
			this.tensor = tensor;
		}
		@Override
		public Matrix zeroCopy(long rows, long cols) {
			return new DenseMatrix(rows, cols);
		}
		@Override
		protected void allocate(long size) {
		}
		@Override
		public Tensor put(long pos, double value) {
			tensor.put(pos, value);
			return this;
		}
		@Override
		public double get(long pos) {
			return tensor.get(pos);
		}
		@Override
		protected Iterator<Long> traverseNonZeroElements() {
			return tensor.traverseNonZeroElements();
		}
		@Override
		public Iterable<Entry<Long, Long>> getNonZeroEntries() {
			return new Iterable<Entry<Long, Long>>() {
				@Override
				public Iterator<Entry<Long, Long>> iterator() {
					return new DenseMatrix.Range2D(0, getRows(), 0, getCols());
				}
			};
		}
	}*/
}
