package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.NoSuchElementException;

import org.junit.Assert;
import org.junit.Test;

import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.SparseSymmetric;
import mklab.JGNN.core.matrix.WrapCols;
import mklab.JGNN.core.matrix.WrapRows;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.core.util.Range2D;

public class MatrixTest {
	public ArrayList<Matrix> allBaseMatrixTypes(long rows, long cols) {
		ArrayList<Matrix> ret = new ArrayList<Matrix>();
		ret.add(new DenseMatrix(rows ,cols));
		ret.add(new SparseMatrix(rows ,cols));
		//create WrapRows
		ArrayList<Tensor> tmp = new ArrayList<Tensor>();
		for(long row=0;row<rows;row++)
			tmp.add(new DenseTensor(cols));
		ret.add(new WrapRows(tmp));
		//create WrapCols
		tmp = new ArrayList<Tensor>();
		for(long col=0;col<cols;col++)
			tmp.add(new DenseTensor(rows));
		ret.add(new WrapCols(tmp));
		return ret;
	}
	
	public ArrayList<Matrix> allSquareMatrixTypes(long dim) {
		ArrayList<Matrix> ret = allBaseMatrixTypes(dim, dim);
		return ret;
	}
	
	public ArrayList<Matrix> allMatrixTypes(long dim) {
		ArrayList<Matrix> ret = allSquareMatrixTypes(dim);
		ret.addAll(allBaseMatrixTypes(dim, dim/2));
		ret.add(new DenseTensor(dim).asRow());
		ret.add(new DenseTensor(dim).asColumn());
		ret.add(new DenseMatrix(dim, 1));
		ret.add(new DenseMatrix(1, dim));
		ret.add(new SparseMatrix(dim, 1));
		ret.add(new SparseMatrix(1, dim));
		ret.add(new SparseTensor(dim).asRow());
		ret.add(new SparseTensor(dim).asColumn());
		return ret;
	}
	@Test
	public void testDoubleConversion() {
		Assert.assertEquals(Matrix.fromDouble(3).size(), 1, 0);
		Assert.assertEquals(Matrix.fromDouble(3).toDouble(), 3, 0);
	}
	@Test
	public void testPut() {
		for(Matrix matrix : allMatrixTypes(6))
			Assert.assertEquals(2.71, matrix.put(0,0,2.71).get(0,0), 0);
		for(Matrix matrix : allMatrixTypes(6))
			Assert.assertEquals(2.71, matrix.put(matrix.getRows()-1,matrix.getCols()-1,2.71)
											.get(matrix.getRows()-1,matrix.getCols()-1), 0);
	}
	@Test
	public void testDimensions() {
		for(Matrix matrix : allBaseMatrixTypes(6,3)) {
			Assert.assertEquals(matrix.getRows(), 6, 0);
			Assert.assertEquals(matrix.getCols(), 3, 0);
			Assert.assertEquals(matrix.size(), 18, 0);
		}
	}
	@Test
	public void testTransposition() {
		for(Matrix matrix : allBaseMatrixTypes(6,3)) 
			Assert.assertEquals(2.71, matrix.put(matrix.getRows()-1,matrix.getCols()-1,2.71)
											.transposed()
											.get(matrix.getCols()-1, matrix.getRows()-1), 0);
		for(Matrix matrix : allMatrixTypes(6)) 
			Assert.assertEquals(2.71, matrix.put(matrix.getRows()-1,matrix.getCols()-1,2.71)
											.asTransposed()
											.get(matrix.getCols()-1, matrix.getRows()-1), 0);
		
	}
	@Test
	public void testAddition() {
		for(Matrix matrix1 : allMatrixTypes(6)) 
			for(Matrix matrix2 : allMatrixTypes(6))
				if(matrix1.isMatching(matrix2)) {
					long row = matrix1.getRows()-1;
					long col = matrix1.getCols()-1;
					Assert.assertEquals(2.02, ((Matrix)matrix1.put(row,col,1.01).add(matrix2.put(row,col,1.01))).get(row,col), 0);
				}
	}
	@Test
	public void testMultiplication() {
		for(Matrix matrix1 : allMatrixTypes(6)) 
			for(Matrix matrix2 : allMatrixTypes(6))
				if(matrix1.getCols()==matrix2.getRows()) {
					long row = matrix1.getRows()-1;
					long col = matrix1.getCols()-1;
					long col2 = matrix2.getCols()-1;
					Assert.assertEquals(0.25, matrix1.put(row,col,0.5).matmul(matrix2.put(col,col2,0.5)).get(row,col2), 0);
				}
	}
	@Test
	public void testMatrixMultiplication() {
		for(Matrix matrix1 : allMatrixTypes(6)) 
			for(Matrix matrix2 : allMatrixTypes(6))
				if(matrix1.isMatching(matrix2) && matrix1.getRows()>2 && matrix1.getCols()>2) {
					matrix1.put(0, 1, 2);
					matrix2.put(2, 1, 3);
					Assert.assertEquals(6, matrix1.matmul(matrix2.asTransposed()).get(0,2), 0);
					Assert.assertEquals(0, matrix1.matmul(matrix2.asTransposed()).get(2,0), 0);
					Assert.assertEquals(6, matrix1.matmul(matrix2, false, true).get(0,2), 0);
					Assert.assertEquals(0, matrix1.matmul(matrix2, false, true).get(2,0), 0);
				}
	}
	@Test
	public void symmetricMatrixShouldWork() {
		Matrix matrix1 = new SparseSymmetric(5, 5)
				.put(3,2,5.3)
				.put(2,3,1.4);
		Assert.assertEquals(6.7, matrix1.get(2,3), 1.E-12);
	}
	@Test
	public void testTransform() {
		for(Matrix matrix : allBaseMatrixTypes(3, 2)) {
			matrix
				.put(0, 0, 7)
				.put(1, 1, 1)
				.put(2, 1, 3);
			Tensor tensor = new DenseTensor(2);
			tensor.put(0, 1);
			tensor.put(1, 2);
			Tensor transformed = matrix.transform(tensor);
			double[] desired = {7., 2., 6.};
			Assert.assertArrayEquals(transformed.toArray(), desired, 0);
		}
	}
	@Test
	public void testDescription() {
		Assert.assertTrue(new SparseMatrix(7, 3).describe().contains("7,3"));
	}
	@Test
	public void testRange2D() {
		long sum = 0;
		for(Entry<Long, Long> element : new Range2D(0,3,0,2))
			sum = sum + element.getKey()*element.getValue();
		Assert.assertEquals(sum, 3, 0);
	}
	@Test(expected = NoSuchElementException.class)
	public void testRange2DOutOfBounds() {
		Iterator<Entry<Long, Long>> range = new Range2D(0,1,0,1);
		range.next();
		range.next();
	}
}
