package mklab.JGNN;

import org.junit.Test;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.distribution.Normal;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.SparseSymmetric;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.core.util.Loss;
import mklab.JGNN.core.util.Range;

import java.lang.reflect.Field;
import java.util.Iterator;
import java.util.NoSuchElementException;

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
	public void tensorShouldPreventInalidDeserialization() {
		Assert.assertEquals(new DenseTensor("").size(), 0, 0);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void tensorShouldPreventInalidDeserialization2() {
		new DenseTensor((String)null);
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
	public void tensorCopyShouldCreateNewDenseTensor() {
		Tensor tensor = (new DenseTensor(10)).setToRandom();
		Tensor originalTensor = tensor.copy();
		Assert.assertNotEquals(originalTensor, tensor);
		Assert.assertEquals(tensor.subtract(originalTensor).abs().sum(), 0, 0);
	}
	
	@Test
	public void tensorCopyShouldCreateNewSparseTensor() {
		Tensor tensor = (new SparseTensor(10)).put(1, 2).put(2, 3);
		Tensor originalTensor = tensor.copy();
		Assert.assertNotEquals(originalTensor, tensor);
		Assert.assertEquals(tensor.subtract(originalTensor).abs().sum(), 0, 0);
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
		Assert.assertSame(tensor.setToUniform(), tensor);
		Assert.assertSame(tensor.setToRandom(new Normal()), tensor);
		Assert.assertSame(tensor.setToProbability(), tensor);
		Assert.assertSame(tensor.selfSqrt(), tensor);
		Assert.assertSame(tensor.selfAbs(), tensor);
		Assert.assertSame(tensor.selfInverse(), tensor);
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
		Assert.assertNotSame(tensor.sqrt(), tensor);
		Assert.assertNotSame(tensor.abs(), tensor);
		Assert.assertNotSame(tensor.inverse(), tensor);
		Assert.assertNotSame(tensor.toProbability(), tensor);
		Assert.assertNotSame(tensor.normalized(), tensor);
	}
	
	@Test
	public void tensorSummaryStatisticsShouldWork() {
		Tensor tensor = Tensor.fromRange(-10, 10);
		Assert.assertEquals(tensor.max(), 9, 0);
		Assert.assertEquals(tensor.min(), -10, 0);
		Assert.assertEquals(tensor.argmax(), 19, 0);
		Assert.assertEquals(tensor.argmin(), 0, 0);
		Assert.assertEquals(new DenseTensor(3).norm(), 0, 0);
		tensor = Tensor.fromRange(1, 11).selfInverse();
		Assert.assertEquals(tensor.argmax(), 0, 0);
		Assert.assertEquals(tensor.max(), 1, 0);
		Assert.assertEquals(tensor.argmin(), 9, 0);
		Assert.assertEquals(tensor.min(), 0.1, 0);	
	}
	
	@Test
	public void dotProductShouldWorkCorrectly() {
		Tensor tensor = new DenseTensor(1, 2);
		Assert.assertEquals(tensor.dot(new DenseTensor(2, 3)), 8, 0);
	}

	@Test
	public void tripleDotProductShouldWorkCorrectly() {
		Tensor tensor = new DenseTensor(1, 2);
		Assert.assertEquals(tensor.dot(new DenseTensor(2, 3), new DenseTensor(5, 3)), 28, 0);
	}
	
	@Test
	public void inverseShouldWorkCorrectly() {
		Tensor tensor = new DenseTensor(10).put(1, 2);
		Assert.assertEquals(tensor.inverse().get(1), 0.5, 0);
		Assert.assertEquals(tensor.inverse().get(0), 0, 0);
		tensor.selfInverse();
		Assert.assertEquals(tensor.get(1), 0.5, 0);
		Assert.assertEquals(tensor.get(0), 0, 0);
	}

	@Test
	public void normalizationShouldWorkCorrectly() {
		Tensor tensor = new DenseTensor(10);
		Assert.assertEquals(tensor.normalized().norm(), 0, 0);
		Assert.assertEquals(tensor.setToNormalized().norm(), 0, 0);
		tensor.setToRandom();
		Assert.assertEquals(tensor.normalized().norm(), 1, 1.E-12);
		Assert.assertEquals(tensor.setToNormalized().norm(), 1, 1.E-12);
	}
	
	@Test
	public void emptyDenseTensorStringShouldBeEmpty() {
		Assert.assertTrue(new DenseTensor().toString().equals(""));
	}

	@Test
	public void emptySparseTensorStringShouldBeEmpty() {
		Assert.assertTrue(new SparseTensor().toString().equals(""));
	}
	
	@Test
	public void conversionToProbabilityShouldSumTo1ExceptOnZeroes() {
		Assert.assertEquals(new DenseTensor(10).setToRandom().toProbability().sum(), 1, 1.E-12);
		Assert.assertEquals(new DenseTensor(10).setToRandom().setToProbability().sum(), 1, 1.E-12);
		Assert.assertEquals(new DenseTensor(10).setToProbability().sum(), 0, 0);
		Assert.assertEquals(new DenseTensor(10).toProbability().sum(), 0, 0);
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
	public void matrixAsTransposedShouldWork() {
		Matrix matrix = new SparseMatrix(5, 5)
				.put(3,2,2.71);
		Assert.assertEquals(2.71, matrix.asTransposed().get(2,3), 0);
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
	
	@Test
	public void tensorDescriptionShouldContainDimensions() {
		Assert.assertTrue(new DenseTensor(71).describe().contains("71"));
	}

	@Test
	public void maxtrixDescriptionShouldContainDimensions() {
		Assert.assertTrue(new SparseMatrix(7, 3).describe().contains("7,3"));
	}
	
	@Test
	public void subtensorShouldAccessCorrectElements() {
		Tensor tensor = Tensor.fromRange(0, 10);
		Assert.assertEquals(tensor.accessSubtensor(1,3).sum(), 3, 0);
		Assert.assertEquals(tensor.accessSubtensor(1).sum(), 45, 0);
	}

	@Test(expected = IllegalArgumentException.class)
	public void overflowDenseTensorPutShouldThrowException() {
		new DenseTensor(3).put(4, 0);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowDenseTensorPutShouldThrowException() {
		new DenseTensor(3).put(-1, 0);
	}
	@Test(expected = IllegalArgumentException.class)
	public void overflowDenseTensorGetShouldThrowException() {
		new DenseTensor(3).get(4);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowDenseTensorGetShouldThrowException() {
		new DenseTensor(3).get(-1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void overflowSparseTensorPutShouldThrowException() {
		new SparseTensor(3).put(4, 0);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowSparseTensorPutShouldThrowException() {
		new SparseTensor(3).put(-1, 0);
	}

	@Test(expected = IllegalArgumentException.class)
	public void overflowSubtensorGetShouldThrowException() {
		new DenseTensor(3).accessSubtensor(0,2).get(3);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowSubensorGetShouldThrowException() {
		new DenseTensor(3).accessSubtensor(1,3).get(-1);
	}
	@Test(expected = IllegalArgumentException.class)
	public void overflowSparseTensorGetShouldThrowException() {
		new SparseTensor(3).get(4);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowSparseTensorGetShouldThrowException() {
		new SparseTensor(3).get(-1);
	}
	@Test(expected = IllegalArgumentException.class)
	public void denseTensorShouldNotAllowNaNPut() {
		new DenseTensor(3).put(0, Double.NaN);
	}
	@Test(expected = IllegalArgumentException.class)
	public void sparseTensorShouldNotAllowNaNPut() {
		new SparseTensor(3).put(0, Double.NaN);
	}
	@Test(expected = IllegalArgumentException.class)
	public void denseTensorShouldNotAllowInfPut() {
		new DenseTensor(3).put(0, Double.POSITIVE_INFINITY);
	}
	@Test(expected = IllegalArgumentException.class)
	public void sparseTensorShouldNotAllowInfPut() {
		new SparseTensor(3).put(0, Double.POSITIVE_INFINITY);
	}
	@Test(expected = RuntimeException.class)
	public void incompatibleTensorsShouldNotBeAdded() {
		new SparseTensor(3).add(new DenseTensor());
	}
	@Test(expected = RuntimeException.class)
	public void wrongTensorSizeShouldNotBeAllowed() {
		new SparseTensor(3).assertSize(4);
	}
	@Test
	public void sparseTensorSetTozeroShouldWork() {
		Assert.assertEquals(new SparseTensor(3).setToRandom().setToZero().norm(), 0, 0);
	}

	@Test(expected = IllegalArgumentException.class)
	public void shouldPreventInvalidSubtensorBounds() {
		new DenseTensor(3).accessSubtensor(3,1);
	}

	@Test(expected = IllegalArgumentException.class)
	public void shouldPreventExcessiveLowerSubtensorBounds() {
		new DenseTensor(3).accessSubtensor(-1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void shouldPreventExcessiveUpperSubtensorBounds() {
		new DenseTensor(3).accessSubtensor(0, 4);
	}
	@Test(expected = IllegalArgumentException.class)
	public void overflowSubtensorPutShouldThrowException() {
		new DenseTensor(3).accessSubtensor(0,2).put(3, 0);
	}
	@Test(expected = IllegalArgumentException.class)
	public void underflowSubensorPutShouldThrowException() {
		new DenseTensor(3).accessSubtensor(1,3).put(-1, 0);
	}

	@Test
	public void shouldBeAbleToCopySubtensor() {
		Assert.assertEquals(new DenseTensor(3).put(1, 2.2).accessSubtensor(1,3).copy().get(0), 2.2, 0);
	}
	
	@Test
	public void subtensorsShouldAccessOriginalElements() {
		Tensor tensor = new DenseTensor(3).setToRandom().setToNormalized();
		String repr = tensor.toString();
		tensor.accessSubtensor(1,3).setToNormalized();
		Assert.assertFalse(repr.equals(tensor.toString()));
	}
	
	
	@Test(expected = IllegalArgumentException.class)
	public void shouldPreventInvalidSubtensorArguments() {
		new AccessSubtensor(null, 1, 3);
	}
	
	@Test
	public void shouldConvertTensorsToColumns() {
		Assert.assertEquals(new SparseTensor(5).putAdd(2, 3).asColumn().get(2, 0), 3, 0);
	}
	@Test
	public void shouldConvertTensorsToRows() {
		Assert.assertEquals(new SparseTensor(5).putAdd(2, 3).asRow().get(0, 2), 3, 0);
	}

	@Test
	public void rangeShouldWork() {
		int result = 0;
		for(long i : new Range(0,4))
			result += i;
		Assert.assertEquals(result, 6, 0);
	}

	@Test(expected = NoSuchElementException.class)
	public void rangeIteratorShouldStopWhenOutOfBounds() {
		Iterator<Long> range = new Range(0,1);
		range.next();
		range.next();
	}

	@Test(expected = RuntimeException.class)
	public void tensorNonFinitenessShouldBeFound() {
		DenseTensor tensor = new DenseTensor(3);
		double[] values = null;
		try {
			Field valueField = DenseTensor.class.getDeclaredField("values");
			valueField.setAccessible(true);
			values = (double[]) valueField.get(tensor);
		}
		catch (Exception e){
			e.printStackTrace();
		}
		values[0] = Double.NaN;
		tensor.assertFinite();
	}
	@Test
	public void tensorFinitenessShouldBeFound() {
		new DenseTensor(3).assertFinite();
	}
	@Test
	public void shouldCountSparseNonZeroElements() { 
		Assert.assertEquals(new SparseTensor(3).put(0, 1).put(1, 0).getNumNonZeroElements(), 1, 0);
	}
	@Test
	public void shouldCountDenseNonZeroElements() { 
		Assert.assertEquals(new DenseTensor(3).setToRandom().add(1).getNumNonZeroElements(), 3, 0);
	}
}
