package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.matrix.AccessRow;
import mklab.JGNN.core.matrix.AccessCol;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.TransposedMatrix;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.core.tensor.DenseTensor;

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
	private String rowName;
	private String colName;
	
	
	protected Matrix(long rows, long cols) {
		init(rows*cols);
		this.rows = rows;
		this.cols = cols;
	}
	
	public final String getRowName() {
		return rowName;
	}
	
	public final String getColName() {
		return colName;
	}
	
	/**
	 * Sets a name for the matrix's row and column dimensions. If set, names are checked for
	 * compatibility during matrix operations.
	 * @param rowName The new row name or <code>null</code> to remove current name.
	 * @param colName The new column name or <code>null</code> to remove current name.
	 * @return <code>this</code> Matrix instance.
	 * @see #getRowName()
	 * @see #getColName()
	 * @see #setRowName(String)
	 * @see #setColName(String)
	 * @see #setDimensionName(String)
	 * @see #setDimensionName(Tensor)
	 */
	public final Matrix setDimensionName(String rowName, String colName) {
		setRowName(rowName);
		setColName(colName);
		return this;
	}
	
	/**
	 * Sets a name for the matrix's row dimension. If set, names are checked for
	 * compatibility during matrix operations.
	 * @param rowName The new row name or <code>null</code> to remove current name.
	 * @return <code>this</code> Matrix instance.
	 * @see #getRowName()
	 * @see #setDimensionName(String, String)
	 * @see #setColName(String)
	 */
	public final Matrix setRowName(String rowName) {
		this.rowName = rowName;
		return this;
	}
	
	/**
	 * Sets a name for the matrix's column dimension. If set, names are checked for
	 * compatibility during matrix operations.
	 * @param colName The new column name or <code>null</code> to remove current name.
	 * @return <code>this</code> Matrix instance.
	 * @see #getColName()
	 * @see #setDimensionName(String, String)
	 * @see #setRowName(String)
	 */
	public final Matrix setColName(String colName) {
		this.colName = colName;
		return this;
	}
	
	/**
	 * Retrieves an iterable that traverses (row, col) entry pairs
	 * of non zero entries.
	 * @return An Entry iterable.
	 * @see #getNonZeroElements()
	 */
	public abstract Iterable<Entry<Long, Long>> getNonZeroEntries();
	
	public final Matrix setDimensionName(Tensor other) {
		super.setDimensionName(other);
		if(rowName==null)
			rowName = other.cast(Matrix.class).getRowName();
		if(colName==null)
			colName = other.cast(Matrix.class).getColName();
		return this;
	}
	
	/**
	 * Creates a Matrix with the same class and dimensions and all element set to zero.
	 * @return A Matrix with the same class and dimensions.
	 * @see #zeroCopy(long, long)
	 */
	@Override
	public final Matrix zeroCopy() {
		return zeroCopy(rows, cols).setDimensionName(this).cast(Matrix.class);
	}
	/**
	 * Creates a Matrix with the same class and dimensions and all element set to zero. This
	 * checks that the copy has a total number of elements equal to the given size.
	 * @param size The desired size of the matrix.
	 * @return A Matrix with the same class and dimensions.
	 * @throws RuntimeException If the resulting tensor is not of the same size.
	 * @see #zeroCopy(long, long)
	 */
	@Override
	public final Tensor zeroCopy(long size) {
		if(size!=size())
			throw new RuntimeException("To avoid ambiguity, desired matrix zeroCopy size "+size+" can only be equal to rows "+rows+" * "+cols);
		return zeroCopy(rows, cols);
	}
	/**
	 * Creates a matrix of the same class and all element set to zero, but with
	 * a given number of rows and columns.
	 * @param rows The number of rows of the matrix.
	 * @param cols The number of columns of the matrix.
	 * @return A Matrix of the same class.
	 * @see #zeroCopy()
	 */
	public abstract Matrix zeroCopy(long rows, long cols);
	/**
	 * Retrieves the number of rows of a matrix.
	 * @return The number of rows.
	 * @see #getCols()
	 * @see #getDimensionSize(String)
	 */
	public final long getRows() {
		return rows;
	}
	/**
	 * Retrieves the number of columns of a matrix.
	 * @return The number of columns.
	 * @see #getRows()
	 * @see #getDimensionSize(String)
	 */
	public final long getCols() {
		return cols;
	}
	/**
	 * Retrieves the value of the dimension with the given name.
	 * @param name The given name.
	 * @return Either the number of rows or the number of cols, depending on which dimension 
	 * 	the given name matches,
	 * @throws RuntimeException if both matrix dimensions have the same name or if the given 
	 *	name is not a matrix dimension.
	 *@see #getRows()
	 *@see #getCols()
	 *@see #setDimensionName(String, String)
	 *@see #setColName(String)
	 *@see #setRowName(String)
	 */
	public final long getDimensionSize(String name) {
		if(rowName!=null && colName!=null && rowName.equals(colName))
			throw new RuntimeException("Cannot call getDimension for the same row and col "
					+ "dimention names in "+describe());
		if(rowName!=null && name.equals(rowName))
			return getRows();
		if(colName!=null && name.equals(colName))
			return getCols();
		throw new RuntimeException("No named dimensions in "+name);
	}
	/**
	 * Retrieves the value stored at a matrix element.
	 * @param row The element's row.
	 * @param col The element's column.
	 * @return The value corresponding to the element (row, col).
	 */
	public final double get(long row, long col) {
		if(row<0 || col<0 || row>=rows || col>=cols)
			throw new IllegalArgumentException("Element ("+row+","+col+") out of range for "+describe());
		return get(row+col*rows);
	}

	/**
	 * Stores values at matrix elements.
	 * @param row The element's row.
	 * @param col The element's column.
	 * @param value The value to store.
	 * @return <code>this</code> Matrix instance.
	 */
	public final Matrix put(long row, long col, double value) {
		if(row<0 || col<0 || row>=rows || col>=cols)
			throw new IllegalArgumentException("Element ("+row+","+col+") out of range for "+describe());
		put(row+col*rows, value);
		return this;
	}
	/**
	 * Creates a transposed copy of the matrix.
	 * Note: Contrary to typical tensor operations, in-place transposition is not supported.
	 * However, related methods can help avoid explicit transposition without allocating more
	 * memory.
	 * @return A transposed copy of the matrix.
	 * @see #matmul(Matrix, boolean, boolean)
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
	private int parallelizedMultiplication = 0;
	/**
	 * Performs the matrix multiplication of <code>this*with</code> and the recipient.
	 * 
	 * @param with The matrix to multiply with.
	 * @return A matrix that stores the outcome of the multiplication.
	 * @see #matmul(Matrix, boolean, boolean)
	 */
	public final Matrix matmul(Matrix with) {
		if(cols!=with.getRows())
			throw new IllegalArgumentException("Mismatched matrix sizes between "+describe()+" and "+with.describe());
		if(colName!=null && with.getRowName()!=null && !colName.equals(with.getRowName()))
			throw new IllegalArgumentException("Mismatched matrix dimension names between "+describe()+" and "+with.describe());
		Matrix ret = determineZeroCopy(with, getRows(), with.getCols(), getCols());
		if(parallelizedMultiplication>1) {
			ArrayList<Entry<Long, Long>> entries = new ArrayList<Entry<Long, Long>>();
			for(Entry<Long, Long> element : getNonZeroEntries()) 
				entries.add(element);
			Thread[] threads = new Thread[parallelizedMultiplication];
			for(int i=0;i<threads.length;i++) {
				int threadId = i;
				threads[i] = new Thread() {
					@Override
					public void run() {
						int start = threadId*entries.size()/threads.length;
						int end = (threadId+1)*entries.size()/threads.length;
						for(int entry=start;entry<end;entry++) {
							Entry<Long, Long> element = entries.get(entry);
							long row = element.getKey();
							long col = element.getValue();
							for(long col2=0;col2<with.getCols();col2++) 
								ret.put(row, col2, ret.get(row, col2) + get(row, col)*with.get(col, col2));
						}
					}
				};
			}
			for(Thread thread : threads)
				thread.start();
			for(Thread thread : threads)
				try {
					thread.join();
				}
				catch (InterruptedException e) {
					e.printStackTrace();
				}
		}
		else {
			if(estimateNumNonZeroElements()/getRows()<with.estimateNumNonZeroElements()/with.getCols()) {
				for(Entry<Long, Long> element : getNonZeroEntries()) {
					long row = element.getKey();
					long col = element.getValue();
					for(long col2=0;col2<with.getCols();col2++) 
						ret.put(row, col2, ret.get(row, col2) + get(row, col)*with.get(col, col2));
				}
			}
			else {
				for(Entry<Long, Long> element : with.getNonZeroEntries()) {
					long row = element.getKey();
					long col = element.getValue();
					for(long row1=0;row1<getRows();row1++) 
						ret.put(row1, col, ret.get(row1, col) + get(row1, row)*with.get(row, col));
				}
			}
		}
		return ret.setRowName(getRowName()).setColName(with.getColName());
	}

	/**
	 * Can be used to perform fast computation of the matrix multiplications 
	 * <br><code>this*with</code>,
	 * <br><code>this.transposed()*with</code>
	 * <br><code>this*with.transposed()</code>, 
	 * <br><code>this.transposed()*with.transposed()</code>
	 * <br>while avoiding the overhead of calling
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
		if((transposeSelf?rowName:colName)!=null &&
				(transposeWith?with.getColName():with.getRowName())!=null &&
				!(transposeSelf?rowName:colName).equals(transposeWith?with.getColName():with.getRowName()))
			throw new IllegalArgumentException("Mismatched matrix dimension names");
		Matrix ret = determineZeroCopy(with, transposeSelf?cols:rows, transposeWith?with.getRows():with.getCols(), transposeWith?with.getCols():with.getRows());
		
		if(parallelizedMultiplication>1) {
			ArrayList<Entry<Long, Long>> entries = new ArrayList<Entry<Long, Long>>();
			for(Entry<Long, Long> element : getNonZeroEntries()) 
				entries.add(element);
			Thread[] threads = new Thread[parallelizedMultiplication];
			for(int i=0;i<threads.length;i++) {
				int threadId = i;
				threads[i] = new Thread() {
					@Override
					public void run() {
						int start = threadId*entries.size()/threads.length;
						int end = (threadId+1)*entries.size()/threads.length;
						for(int entry=start;entry<end;entry++) {
							Entry<Long, Long> element = entries.get(entry);
							long row = transposeSelf?element.getValue():element.getKey();
							long col = transposeSelf?element.getKey():element.getValue();
							for(long col2=0;col2<(transposeWith?with.getRows():with.getCols());col2++) { 
								double val = get(element.getKey(),element.getValue())*with.get(transposeWith?col2:col, transposeWith?col:col2);
								ret.put(row, col2, val + ret.get(row, col2));
								
							}
						}
					}
				};
			}
			for(Thread thread : threads)
				thread.start();
			for(Thread thread : threads)
				try {
					thread.join();
				}
				catch (InterruptedException e) {
					e.printStackTrace();
				}
		}
		else {
			if(estimateNumNonZeroElements()/(transposeSelf?getCols():getRows())<with.estimateNumNonZeroElements()/(transposeWith?with.getRows():with.getCols())) {
				for(Entry<Long, Long> element : getNonZeroEntries()) {
					long row = transposeSelf?element.getValue():element.getKey();
					long col = transposeSelf?element.getKey():element.getValue();
					for(long col2=0;col2<(transposeWith?with.getRows():with.getCols());col2++) 
						ret.put(row, col2, ret.get(row, col2) + get(element.getKey(),element.getValue())*with.get(transposeWith?col2:col, transposeWith?col:col2));
				}
			}
			else {
				for(Entry<Long, Long> element : with.getNonZeroEntries()) {
					long row = transposeWith?element.getValue():element.getKey();
					long col = transposeWith?element.getKey():element.getValue();
					for(long row1=0;row1<(transposeSelf?getCols():getRows());row1++) 
						ret.put(row1, col, ret.get(row1, col) + get(transposeSelf?row:row1, transposeSelf?row1:row)*with.get(element.getKey(), element.getValue()));
				}
			}
		}
		return ret.setRowName(transposeSelf?getColName():getRowName()).setColName(transposeWith?with.getRowName():with.getColName());
	}
	
	/**
	 * Produces the external product of two tensors.
	 * This is equivalent but faster to calling matmul(horizontal.asColum(), vertical.asRow()).
	 * @param horizontal The first tensor.
	 * @param vertical The second tensor.
	 * @return A dense matrix holding the external product.
	 */
	public static Matrix external(Tensor horizontal, Tensor vertical) {
		Matrix ret = new DenseMatrix(horizontal.size(), vertical.size());
		for(long row=0;row<horizontal.size();row++)
			for(long col=0;col<vertical.size();col++) 
				ret.put(row, col, horizontal.get(row)*vertical.get(col));
		return ret;
	}
	
	/*@Override
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
	}*/
	
	@Override
	protected boolean isMatching(Tensor other) {
		if(!(other instanceof Matrix)) {
			if(rows!=1 && cols!=1)
				return false;
			else
				return super.isMatching(other);
		}
		Matrix otherMatrix = other.cast(Matrix.class);
		if(rows!=otherMatrix.rows || cols!=otherMatrix.cols)
			return false;
		if(rowName!=null && otherMatrix.rowName!=null && !rowName.equals(otherMatrix.rowName))
			return false;
		if(colName!=null && otherMatrix.colName!=null && !colName.equals(otherMatrix.colName))
			return false;
		return true;
	}
	
	@Override
	public String describe() {
		return getClass().getSimpleName()+" ("+(rowName==null?"":(rowName+" "))+rows+","+(colName==null?"":(" "+colName+" "))+cols+")";
	}
	
	/**
	 * Produces a mask that indicates the non-zero elements of the matrix.
	 * Element's correspond to the matrix's whose non-zero ones are set to 1.
	 * @return A matrix of the same dimensions.
	 */
	public Matrix onesMask() {
		Matrix ones = zeroCopy(getRows(), getCols());
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			long row = element.getKey();
			long col = element.getValue();
			if(get(row, col)!=0)
				ones.put(row, col, 1.);
		}
		return ones;
	}
	
	/**
	 * Creates a copy of the Matrix that holds its symmetrically normalized version.
	 * @return A new Matrix of the same dimensions.
	 * @see #setToSymmetricNormalization()
	 */
	public Matrix symmetricNormalization() {
		return ((Matrix)copy()).setToSymmetricNormalization();
	}

	/**
	 * Sets the matrix's specified main diagonal elements to a given value value.
	 * @param value The value to set to the diagonal's elements.
	 * @return <code>this</code> Matrix instance.
	 * @see #setDiagonal(long, double)
	 */
	public Matrix setMainDiagonal(double value) {
		return setDiagonal(0, value);
	}
	
	/**
	 * Sets the matrix's specified diagonal elements to a given value.
	 * @param diagonal Which diagonal to set. 0 is the main diagonal
	 * @param value The value to set to the diagonal's elements.
	 * @return <code>this</code> Matrix instance.
	 * @see #setMainDiagonal(double)
	 */
	public Matrix setDiagonal(long diagonal, double value) {
		if(cols>rows) {
			for(long col=Math.max(0, diagonal);col<Math.min(cols, cols+diagonal);col++)
				if(col<rows)
					put(col, col, value);
		}
		else {
			for(long row=Math.max(0, diagonal);row<Math.min(rows, rows+diagonal);row++)
				if(row<cols)
					put(row, row, value);
		}
		return this;
	}
	
	/**
	 * Sets the Matrix to its symmetrically normalized transformation 
	 * by appropriately adjusting its element values.
	 * @return <code>this</code> Matrix instance.
	 * @see #symmetricNormalization()
	 */
	public Matrix setToSymmetricNormalization() {
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
	 * Retrieves either the given row or column as a trensor.
	 * @param index The dimension index to access.
	 * @param name The dimension's name.
	 * @return Either a {@link AccessRow} or a {@link AccessCol} at the given index.
	 * @see #accessRow(long)
	 * @see #accessCol(long)
	 */
	public Tensor accessDim(long index, String name) {
		if(rowName!=null && colName!=null && rowName.equals(colName))
			throw new RuntimeException("Cannot call accessDim for the same row and col "
					+ "dimension names in "+describe());
		if(rowName!=null && name.equals(rowName))
			return accessRow(index);
		if(colName!=null && name.equals(colName))
			return accessCol(index);
		throw new RuntimeException("No named dimensions in "+name);
	}
	
	/**
	 * Retrieves the given row as a tensor. Editing the result
	 * also edits the original matrix.
	 * No new memory is allocated for matrix values.
	 * @param row The given row.
	 * @return An {@link AccessRow} instance of the corresponding row.
	 * @see #accessDim(long, String)
	 * @see #accessCol(long)
	 * @see #accessRows()
	 * @see #accessRows(long...)
	 * @see #accessRows(Tensor)
	 */
	public Tensor accessRow(long row) {
		return new AccessRow(this, row);
	}

	/**
	 * Retrieves the given column as a tensor. Editing the result
	 * also edits the original matrix.
	 * No new memory is allocated for matrix values.
	 * @param col The given column.
	 * @return An {@link AccessCol} of the corresponding column.
	 * @see #accessDim(long, String)
	 * @see #accessRow(long)
	 * @see #accessColumns()
	 * @see #accessColumns(long...)
	 * @see #accessColumns(Tensor)
	 */
	public Tensor accessCol(long col) {
		return new AccessCol(this, col);
	}
	/*
	@Override
	public String toString() {
		StringBuilder res = new StringBuilder();
		for(long row=0;row<rows;row++) {
			res.append("[");
			for(long col=0;col<cols;col++) {
				if(col!=0)
					res.append(",");
				res.append(get(row, col));
			}
			res.append("]");
		}
		return ""+res.toString();
	}*/
	
	@Override
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
	}
	

	public String toNonZeroString() {
		String ret = "";
		for(Entry<Long, Long> element : getNonZeroEntries()) {
			double value = get(element.getKey(), element.getValue());
			if(value!=0)
				ret += element.getKey()+", "+element.getValue()+": "+value+"\n";
		}
		return ret;
	}
	
	/*public final static Matrix fromColumns(List<Tensor> tensors) {
		Matrix ret = new SparseMatrix(tensors.get(0).size(), tensors.size());
		for(int col=0;col<tensors.size();col++) 
			for(long row : tensors.get(col).getNonZeroElements())
				ret.put(row, col, tensors.get(col).get(row));
		return ret;
	}*/
	/**
	 * Organizes matrix rows to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @return A list of {@link AccessRow} instances.
	 * @see #accessRow(long)
	 * @see #accessRows(long...)
	 * @see #accessRows(Tensor)
	 * @see #accessRows(Iterable)
	 * @see #accessColumns()
	 */
	public final List<Tensor> accessRows() {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long row=0;row<getRows();row++)
			ret.add(accessRow(row));
		return ret;
	}
	/**
	 * Organizes specific matrix columns to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @return A list of {@link AccessCol} instances.
	 * @see #accessCol(long)
	 * @see #accessColumns(long...)
	 * @see #accessColumns(Tensor)
	 * @see #accessColumns(Iterable)
	 * @see #accessRows()
	 */
	public final List<Tensor> accessColumns() {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long col=0;col<getCols();col++)
			ret.add(accessCol(col));
		return ret;
	}
	
	/**
	 * Organizes specific matrix rows to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param rows An array of rows to access.
	 * @return A list of {@link AccessRow} instances.
	 * @see #accessRow(long)
	 * @see #accessRows()
	 * @see #accessRows(Tensor)
	 * @see #accessRows(Iterable)
	 * @see #accessColumns(long...)
	 */
	public final Matrix accessRows(long ... rows) {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long row : rows)
			ret.add(accessRow(row));
		return new WrapRows(ret);
	}
	/**
	 * Organizes specific matrix columns to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param cols An array of columns to access.
	 * @return A list of {@link AccessCol} instances.
	 * @see #accessCol(long)
	 * @see #accessColumns()
	 * @see #accessColumns(Tensor)
	 * @see #accessColumns(Iterable)
	 * @see #accessRows(long...)
	 */
	public final Matrix accessColumns(long ... cols) {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long col=0;col<getCols();col++)
			ret.add(accessCol(col));
		return new WrapCols(ret);
	}

	/**
	 * Organizes specific matrix rows to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param rows A tensor whose values hold the rows to access.
	 * @return A list of {@link AccessRow} instances.
	 * @see #accessRow(long)
	 * @see #accessRows(long...)
	 * @see #accessRows()
	 * @see #accessRows(Iterable)
	 * @see #accessColumns(Tensor)
	 */
	public final Matrix accessRows(Tensor rows) {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long row=0;row<rows.size();row++)
			ret.add(accessRow((long)rows.get(row)));
		return new WrapRows(ret);
	}
	/**
	 * Organizes matrix columns to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param cols A tensor whose values hold the columns to access.
	 * @return A list of {@link AccessCol} instances.
	 * @see #accessCol(long)
	 * @see #accessColumns(long...)
	 * @see #accessColumns()
	 * @see #accessColumns(Iterable)
	 * @see #accessRows(Tensor)
	 */
	public final Matrix accessColumns(Tensor cols) {
		ArrayList<Tensor> ret = new ArrayList<Tensor>();
		for(long col=0;col<cols.size();col++)
			ret.add(accessCol((long)cols.get(col)));
		return new WrapCols(ret);
	}

	/**
	 * Organizes some matrix rows to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param rowIds The rows to access.
	 * @return A list of {@link AccessRow} instances.
	 * @see #accessRow(long)
	 * @see #accessRows(long...)
	 * @see #accessRows(Tensor)
	 * @see #accessRows()
	 * @see #accessColumns(Iterable)
	 */
	public final List<Tensor> accessRows(Iterable<Long> rowIds) {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long row : rowIds)
			ret.add(accessRow(row));
		return ret;
	}
	
	/**
	 * Organizes some matrix columns to a list of tensors that share entries.
	 * This operation does not allocate memory for matrix elements and editing 
	 * tensor elements edits the original matrix's elements.
	 * @param colIds The columns to access.
	 * @return A list of {@link AccessCol} instances.
	 * @see #accessCol(long)
	 * @see #accessColumns(long...)
	 * @see #accessColumns(Tensor)
	 * @see #accessColumns()
	 * @see #accessRows(Iterable)
	 */
	public final List<Tensor> accessColumns(Iterable<Long> colIds) {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long col : colIds)
			ret.add(accessCol(col));
		return ret;
	}
	/*public final List<Tensor> toSparseColumns() {
		List<Tensor> ret = new ArrayList<Tensor>();
		for(long col=0;col<getCols();col++)
			ret.add(new SparseTensor(getRows()));
		for(Entry<Long, Long> entry : getNonZeroEntries()) {
			long row = entry.getKey();
			long col = entry.getValue();
			ret.get((int)col).put(row, get(row, col));
		}
		return ret;
	}*/
	
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
	
	protected Matrix determineZeroCopy(Matrix with, long rows, long cols, long intermediate) {
		if(1-Math.pow(1-density()*with.density(), intermediate) < 0.3)
			return new SparseMatrix(rows, cols);
		/*if(with instanceof SparseMatrix) 
			return ((Matrix)with).zeroCopy(rows, cols);
		try {
			return zeroCopy(rows, cols);
		}
		catch(UnsupportedOperationException e) {
		}
		try {
			return ((Matrix)with).zeroCopy(rows, cols);
		}
		catch(UnsupportedOperationException e) {
		}*/
		return new DenseMatrix(rows, cols);
		//throw new UnsupportedOperationException("Neither "+describe()+" nor "+with.describe()+" support zeroCopy("+rows+", "+cols+")");
	}
	/**
	 * Creates a copy of the matrix organized as a dense matrix.
	 * @return A {@link DenseMatrix} instance.
	 */
	public DenseMatrix toDense() {
		return (DenseMatrix)new DenseMatrix(getRows(), getCols()).selfAdd(this).setDimensionName(this);
	}
	/**
	 * Creates a copy of the matrix organized as a sparse matrix.
	 * @return A {@link SparseMatrix} instance.
	 */
	public SparseMatrix toSparse() {
		return (SparseMatrix)new SparseMatrix(getRows(), getCols()).selfAdd(this).setDimensionName(this);
	}
	/**
	 * Creates a sparse unit matrix.
	 * @param dims The dimensions of the unit matrix.
	 * @return A sparse matrix.
	 */
	public static SparseMatrix eye(long dims) {
		SparseMatrix matrix = new SparseMatrix(dims, dims);
		for(long pos=0;pos<dims;pos++)
			matrix.put(pos, pos, 1);
		return matrix;
	}
}
