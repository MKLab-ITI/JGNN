package mklab.JGNN.core.matrix;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.VectorizedTensor;
import mklab.JGNN.core.util.Range2D;

/**
 * Implements a dense {@link Matrix} where all elements are stored in memory.
 * For matrices with more than MAXINT number of elements or many zeros use the {@link SparseMatrix}
 * structure.
 * 
 * @author Emmanouil Krasanakis
 */
public class VectorizedMatrix extends Matrix {
	public VectorizedTensor tensor;
	/**
	 * Generates a dense matrix with the designated number of rows and columns.
	 * @param rows The number of rows.
	 * @param cols The number of columns.
	 */
	public VectorizedMatrix(long rows, long cols) {
		super(rows, cols);
	}
	@Override
	public Matrix zeroCopy(long rows, long cols) {
		if(rows<=100000/cols)
			return new DenseMatrix(rows, cols).setDimensionName(getRowName(), getColName());
		return new VectorizedMatrix(rows, cols).setDimensionName(getRowName(), getColName());
	}
	@Override
	protected void allocate(long size) {
		tensor = new VectorizedTensor(size);
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
	public Iterator<Long> traverseNonZeroElements() {
		return tensor.traverseNonZeroElements();
	}
	@Override
	public Iterable<Entry<Long, Long>> getNonZeroEntries() {
		return new Range2D(0, getRows(), 0, getCols());
	}
	@Override
	public void release() {
		tensor.release();
	}
	@Override
	public void persist() {
		tensor.persist();
	}
	

	@Override
	public Matrix matmul(Matrix with) {
	    if (with instanceof SparseMatrix)
	        return super.matmul(with);
		if(getCols()!=with.getRows()) 
			throw new IllegalArgumentException("Mismatched matrix sizes between "+describe()+" and "+with.describe());
		if(getColName()!=null && with.getRowName()!=null && !getColName().equals(with.getRowName()))
			throw new IllegalArgumentException("Mismatched matrix dimension names between "+describe()+" and "+with.describe());
		VectorizedMatrix ret = new VectorizedMatrix(getRows(), with.getCols());
	    double[] with_tensor_values = (with instanceof VectorizedMatrix)
				?((VectorizedMatrix) with).tensor.values
				:((DenseMatrix) with).tensor.values;
	
		int rows = (int) getRows();
		int cols = (int) getCols();
		int withRows = (int) with.getRows();
		int withCols = (int) with.getCols();
		for(int col2=0;col2<withCols;++col2) 
			for(int row=0;row<rows;++row) {
				int resultIndex = row+col2*rows;
				double summand = 0;
				for(int col=0;col<cols;++col)  {
					int thisIndex = row+col*rows;
					int withIndex = col+col2*withRows;
					summand += tensor.values[thisIndex]*with_tensor_values[withIndex];
				}
				ret.tensor.values[resultIndex] = summand;
			}
		return ret.setRowName(getRowName()).setColName(with.getColName());
	}
	

	@Override
	public Matrix matmul(Matrix with, boolean transposeThis, boolean transposeWith) {
	    if (with instanceof SparseMatrix)
	        return super.matmul(with, transposeThis, transposeWith);
	    
	    // Determine the dimensions based on whether we transpose or not
	    int rowsThis = (int) (transposeThis ? getCols() : getRows());
	    int colsThis = (int) (transposeThis ? getRows() : getCols());
	    int rowsWith = (int) (transposeWith ? with.getCols() : with.getRows());
	    int colsWith = (int) (transposeWith ? with.getRows() : with.getCols());

	    if(colsThis!=rowsWith)
			throw new IllegalArgumentException("Mismatched matrix sizes");
		if((transposeThis?getRowName():getColName())!=null &&
				(transposeWith?with.getColName():with.getRowName())!=null &&
				!(transposeThis?getRowName():getColName()).equals(transposeWith?with.getColName():with.getRowName()))
			throw new IllegalArgumentException("Mismatched matrix dimension names");
		
	 
	    // Create the resulting matrix
	    VectorizedMatrix ret = new VectorizedMatrix(rowsThis, colsWith);
	    double[] with_tensor_values = (with instanceof VectorizedMatrix)
	    											?((VectorizedMatrix) with).tensor.values
	    											:((DenseMatrix) with).tensor.values;
	    
	    for (int col2 = 0; col2 < colsWith; ++col2) {
	        for (int row = 0; row < rowsThis; ++row) {
	            int resultIndex = row + col2 * rowsThis;
	            double summand = 0;
	            for (int col = 0; col < colsThis; ++col) {
	                int thisIndex = transposeThis ? col + row * colsThis : row + col * rowsThis;
	                int withIndex = transposeWith ? col2 + col * colsWith : col + col2 * rowsWith;
	                summand += tensor.values[thisIndex] * with_tensor_values[withIndex];
	            }
	            ret.tensor.values[resultIndex] = summand;
	        }
	    }

	    return ret
	    		.setRowName(transposeThis?getColName():getRowName())
	    		.setColName(transposeWith?with.getRowName():with.getColName());
	}

}