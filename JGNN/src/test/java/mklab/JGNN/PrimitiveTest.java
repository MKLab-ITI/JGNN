package mklab.JGNN;

import org.junit.Test;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.SparseSymmetric;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Loss;

import org.junit.Assert;

public class PrimitiveTest {
	
	@Test
	public void lossShouldBeFiniteOnEdgeCases() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 1)));
	}

	@Test
	public void lossDerivativeShouldBeFiniteOnEdgeCases() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 1)));
	}

	@Test(expected = RuntimeException.class)
	public void lossShouldThrowExceptionOnInvalidLabel() {
		Loss.crossEntropyDerivative(0, 0.1);
	}
	
	@Test(expected = RuntimeException.class)
	public void lossShouldThrowExceptionOnNegativeLabel() {
		Loss.crossEntropyDerivative(0, -1);
	}
	
	@Test(expected = RuntimeException.class)
	public void lossShouldThrowExceptionOnNegativeOutput() {
		Loss.crossEntropyDerivative(-1, 0);
	}
	
	@Test(expected = RuntimeException.class)
	public void lossShouldThrowExceptionOnOutOfBoundsOutput() {
		Loss.crossEntropyDerivative(2, 0);
	}
	
	@Test
	public void tensorsShouldHaveCorrectDimensions() {
		Assert.assertEquals((new DenseTensor(10)).size(), 10);
	}
	
	@Test
	public void tensorShouldSerialize() {
		Tensor tensor = new DenseTensor(10);
		Assert.assertEquals(tensor.toString().length(), 4*10-1);
	}
	
	@Test
	public void tensorShouldBeReconstructableFromSerialization() {
		Tensor tensor = new DenseTensor(10);
		String originalTensor = tensor.toString();
		String newTensor = (new DenseTensor(originalTensor)).toString();
		Assert.assertEquals(originalTensor, newTensor);
	}
	
	@Test
	public void tensorRandomizeShouldSetNewValues() {
		Tensor tensor = new DenseTensor(10);
		String zeroString = tensor.toString();
		tensor.setToRandom();
		Assert.assertTrue(!zeroString.equals(tensor.toString()));
	}
	
	@Test
	public void tensorZeroCopyShouldCreateNewZeroTensor() {
		Tensor tensor = (new DenseTensor(10)).setToRandom();
		String originalTensor = tensor.toString();
		tensor.zeroCopy();
		Assert.assertEquals(originalTensor, tensor.toString());
	}
	
	@Test
	public void tensorMultiplicationWithZeroShouldBeZero() {
		Tensor tensor = new DenseTensor(10);
		String zeroString = tensor.toString();
		tensor.setToRandom().selfMultiply(0);
		Assert.assertEquals(zeroString, tensor.toString());
	}
	
	@Test
	public void tensorSelfOperationsShouldYieldSelf() {
		Tensor tensor = new DenseTensor(10);
		Assert.assertSame(tensor.setToNormalized(), tensor);
		Assert.assertSame(tensor.setToRandom(), tensor);
		Assert.assertSame(tensor.setToOnes(), tensor);
		Assert.assertSame(tensor.setToUniform(), tensor);
		Assert.assertSame(tensor.setToZero(), tensor);
		Assert.assertSame(tensor.selfAdd(new DenseTensor(10)), tensor);
		Assert.assertSame(tensor.selfMultiply(new DenseTensor(10)), tensor);
		Assert.assertSame(tensor.selfSubtract(new DenseTensor(10)), tensor);
		Assert.assertSame(tensor.selfMultiply(0), tensor);
	}
	
	@Test
	public void tensorPairOperationsShouldYieldNewTensor() {
		Tensor tensor = new DenseTensor(10);
		Assert.assertNotSame(tensor.normalized(), tensor);
		Assert.assertNotSame(tensor.zeroCopy(), tensor);
		Assert.assertNotSame(tensor.add(new DenseTensor(10)), tensor);
		Assert.assertNotSame(tensor.multiply(new DenseTensor(10)), tensor);
		Assert.assertNotSame(tensor.subtract(new DenseTensor(10)), tensor);
		Assert.assertNotSame(tensor.multiply(0), tensor);
	}

	@Test
	public void matrixShouldHoldValues() {
		Matrix matrix = new SparseMatrix(5, 5)
				.put(3,2,2.71);
		Assert.assertEquals(2.71, matrix.get(3,2), 0);
	}
	
	@Test
	public void matrixTranspositionShouldWork() {
		Matrix matrix = new SparseMatrix(5, 5)
				.put(3,2,2.71);
		Assert.assertEquals(2.71, matrix.transposed().get(2,3), 0);
	}

	@Test
	public void matrixAdditionShouldWork() {
		Matrix matrix1 = new SparseMatrix(5, 5)
				.put(3,2,1.01);
		Matrix matrix2 = new DenseMatrix(5, 5)
				.put(2,3,1.7);
		Assert.assertEquals(2.71, ((Matrix)matrix1.add(matrix2.transposed())).get(3,2), 0);
	}

	@Test
	public void symmetricMatrixShouldWork() {
		Matrix matrix1 = new SparseSymmetric(5, 5)
				.put(3,2,5.3)
				.put(2,3,1.4);
		Assert.assertEquals(6.7, matrix1.get(2,3), 1.E-12);
	}


	@Test
	public void matrixTransformShouldWorkCorrectly() {
		Matrix matrix = new SparseMatrix(3, 2)
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
	
	@Test
	public void zeroTensorShouldBeNormalizeable() {
		(new DenseTensor(10)).normalized();
	}
}
