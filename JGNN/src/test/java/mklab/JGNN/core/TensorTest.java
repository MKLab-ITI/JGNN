package mklab.JGNN.core;

import org.junit.Test;

import mklab.JGNN.core.distribution.Normal;
import mklab.JGNN.core.tensor.AccessSubtensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.RepeatTensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.core.util.Range;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.NoSuchElementException;

import org.junit.Assert;

public class TensorTest {
	public ArrayList<Tensor> allTensorTypes() {
		ArrayList<Tensor> tensorTypeInstances = new ArrayList<Tensor>();
		tensorTypeInstances.add(new DenseTensor());
		tensorTypeInstances.add(new SparseTensor());
		return tensorTypeInstances;
	}
	public ArrayList<Tensor> allTensorTypes(long size) {
		ArrayList<Tensor> tensorTypeInstances = new ArrayList<Tensor>();
		tensorTypeInstances.add(new DenseTensor(size));
		tensorTypeInstances.add(new SparseTensor(size));
		tensorTypeInstances.add(new DenseTensor(size+2).accessSubtensor(1, size+1));
		tensorTypeInstances.add(new SparseTensor(size+2).accessSubtensor(1, size+1));
		return tensorTypeInstances;
	}
	public ArrayList<Tensor> allTensorTypesRandom(long size) {
		ArrayList<Tensor> tensorTypeInstances = allTensorTypes(size);
		for(Tensor tensor : tensorTypeInstances)
			tensor.setToRandom(new Normal());
		return tensorTypeInstances;
	}
	@Test
	public void testDoubleConversion() {
		Assert.assertEquals(Tensor.fromDouble(3).size(), 1, 0);
		Assert.assertEquals(Tensor.fromDouble(3).toDouble(), 3, 0);
	}
	@Test
	public void testTensorDimensions() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertEquals(tensor.size(), 10);
			Assert.assertTrue(tensor.describe().contains("10"));
		}
	}
	@Test
	public void testDenseTensorSerialization() {
		// no serialization for other types of tensors
		Assert.assertEquals(new DenseTensor("").size(), 0, 0);
		Tensor tensor = new DenseTensor(10);
		String originalTensor = tensor.toString();
		String newTensor = (new DenseTensor(originalTensor)).toString();
		Assert.assertEquals(originalTensor.toString().length(), 4*10-1);
		Assert.assertEquals(originalTensor, newTensor);
	}
	@Test(expected = IllegalArgumentException.class)
	public void testNullTensorDeerialization() {
		new DenseTensor((String)null);
	}	
	@Test
	public void testSetToRandom() {
		for(Tensor tensor : allTensorTypes(10)) {
			String zeroString = tensor.toString();
			tensor.setToRandom();
			Assert.assertFalse(zeroString.equals(tensor.toString()));
		}
		for(Tensor tensor : allTensorTypes(10)) {
			String zeroString = tensor.toString();
			tensor.setToRandom(new Normal());
			Assert.assertFalse(zeroString.equals(tensor.toString()));
		}
	}
	
	@Test
	public void testZeroCopy() {
		for(Tensor tensor : allTensorTypesRandom(10)) {
			String originalTensorString = tensor.toString();
			Tensor zeroCopy = tensor.zeroCopy();
			Assert.assertEquals(tensor.describe(), zeroCopy.describe());
			Assert.assertNotEquals(zeroCopy, tensor);
			Assert.assertNotEquals(originalTensorString, zeroCopy.toString());
			Assert.assertEquals(zeroCopy.abs().sum(), 0, 0);
			Assert.assertEquals(zeroCopy.norm(), 0, 0);
		}
	}

	@Test
	public void testCopy() {
		for(Tensor tensor : allTensorTypesRandom(10)) {
			Tensor copy = tensor.copy();
			Assert.assertEquals(tensor.describe(), copy.describe());
			Assert.assertEquals(tensor.toString(), copy.toString());
			Assert.assertEquals(tensor.subtract(copy).abs().sum(), 0, 0);
			Assert.assertEquals(tensor.subtract(copy).norm(), 0, 0);
		}
	}
	
	@Test
	public void testMultiplicationWithZero() {
		for(Tensor tensor : allTensorTypesRandom(10)) {
			Assert.assertEquals(tensor.multiply(0).abs().sum(), 0, 0);
			Assert.assertEquals(tensor.multiply(0).norm(), 0, 0);
			tensor.selfMultiply(0);
			Assert.assertEquals(tensor.abs().sum(), 0, 0);
			Assert.assertEquals(tensor.norm(), 0, 0);
		}
	}
	
	@Test
	public void testSelfOperations() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertSame(tensor.setToNormalized(), tensor);
			Assert.assertSame(tensor.setToRandom(), tensor);
			Assert.assertSame(tensor.setToOnes(), tensor);
			Assert.assertSame(tensor.setToUniform(), tensor);
			Assert.assertSame(tensor.setToZero(), tensor);
			Assert.assertSame(tensor.selfAdd(new DenseTensor(10)), tensor);
			Assert.assertSame(tensor.selfAdd(1), tensor);
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
	}
	
	@Test
	public void testNewTensorCreation() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertNotSame(tensor.normalized(), tensor);
			Assert.assertNotSame(tensor.zeroCopy(), tensor);
			Assert.assertNotSame(tensor.add(1), tensor);
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
	}
	
	@Test
	public void testSummaryStatistics() {
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
	public void testDot() {
		Assert.assertEquals(new DenseTensor(1, 2).dot(new DenseTensor(2, 3)), 8, 0);
	}

	@Test
	public void testTripleDot() {
		Tensor tensor = new DenseTensor(1, 2);
		Assert.assertEquals(tensor.dot(new DenseTensor(2, 3), new DenseTensor(5, 3)), 28, 0);
	}
	
	@Test
	public void testInverse() {
		for(Tensor tensor : allTensorTypes(10)) {
			tensor.put(1, 2);
			Assert.assertEquals(tensor.inverse().get(1), 0.5, 0);
			Assert.assertEquals(tensor.inverse().get(0), 0, 0);
			tensor.selfInverse();
			Assert.assertEquals(tensor.get(1), 0.5, 0);
			Assert.assertEquals(tensor.get(0), 0, 0);
		}
	}

	@Test
	public void testNormalization() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertEquals(tensor.normalized().norm(), 0, 0);
			Assert.assertEquals(tensor.setToNormalized().norm(), 0, 0);
			tensor.setToRandom();
			Assert.assertEquals(tensor.normalized().norm(), 1, 1.E-12);
			Assert.assertEquals(tensor.setToNormalized().norm(), 1, 1.E-12);
		}
	}
	
	@Test
	public void testToString() {
		for(Tensor tensor : allTensorTypes()) 
			Assert.assertTrue(tensor.toString().equals(""));
	}
	
	@Test
	public void testToProbability() {
		for(Tensor tensor : allTensorTypesRandom(10)) {
			Assert.assertEquals(tensor.toProbability().sum(), 1, 1.E-12);
			Assert.assertEquals(tensor.setToProbability().sum(), 1, 1.E-12);
		}
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertEquals(tensor.toProbability().sum(), 0, 0);
			Assert.assertEquals(tensor.setToProbability().sum(), 0, 0);
		}
	}
	
	@Test
	public void subtensorShouldAccessCorrectElements() {
		Tensor tensor = Tensor.fromRange(0, 10);
		Assert.assertEquals(tensor.accessSubtensor(1,3).sum(), 3, 0);
		Assert.assertEquals(tensor.accessSubtensor(1).sum(), 45, 0);
	}
	
	@Test
	public void testPutNan() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.put(1, Double.NaN);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testPutInfinity() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.put(1, Double.POSITIVE_INFINITY);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testPutOverflow() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.put(11, 0);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}
	
	@Test
	public void testPutUnderflow() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.put(-1, 0);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testGetOverflow() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.get(11);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testGetUnderflow() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.get(-1);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testAssertSize() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.assertSize(9);
				Assert.fail();
			}
			catch(RuntimeException e) {
			}
		}
	}

	@Test
	public void testIncompatibleOperation() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.add(new DenseTensor(3));
				Assert.fail();
			}
			catch(RuntimeException e) {
			}
		}
	}

	@Test
	public void testSubtensorWrongStart() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.accessSubtensor(-1);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testSubtensorWrongEnd() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.accessSubtensor(0, 11);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testSubtensorWrongRange() {
		for(Tensor tensor : allTensorTypes(10)) {
			try {
				tensor.accessSubtensor(3, 1);
				Assert.fail();
			}
			catch(IllegalArgumentException e) {
			}
		}
	}

	@Test
	public void testSubtensorCopy() {
		for(Tensor tensor : allTensorTypes(10))
			Assert.assertEquals(tensor.put(1, 2.2).accessSubtensor(1,3).copy().get(0), 2.2, 0);
	}
	
	@Test
	public void testSubtensorOriginalAccess() {
		for(Tensor tensor : allTensorTypesRandom(10)) {
			String repr = tensor.toString();
			tensor.accessSubtensor(1,3).setToNormalized();
			Assert.assertNotEquals(repr, tensor.toString());
		}
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testInvalidSubtensorConstructorArgument() {
		new AccessSubtensor(null, 1, 3);
	}
	
	@Test
	public void testAsColum() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertEquals(tensor.putAdd(2, 3).asColumn().get(2, 0), 3, 0);
			Assert.assertEquals(tensor.asColumn().getCols(), 1, 0);
		}
	}
	@Test
	public void testAsRow() {
		for(Tensor tensor : allTensorTypes(10)) {
			Assert.assertEquals(tensor.putAdd(2, 3).asRow().get(0, 2), 3, 0);
			Assert.assertEquals(tensor.asRow().getRows(), 1, 0);
		}
	}

	@Test
	public void testRange() {
		int result = 0;
		for(long i : new Range(0,4))
			result += i;
		Assert.assertEquals(result, 6, 0);
	}

	@Test(expected = NoSuchElementException.class)
	public void testRangeOutOfBounds() {
		Iterator<Long> range = new Range(0,1);
		range.next();
		range.next();
	}

	@Test(expected = RuntimeException.class)
	public void testAssertNotFinite() {
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
	public void testAssertFinite() {
		new DenseTensor(3).assertFinite();
	}
	@Test
	public void testOneHotNumNonZero() { 
		for(Tensor tensor : allTensorTypes(10)) 
			Assert.assertEquals(tensor.put(0, 1).put(1, 0).estimateNumNonZeroElements(), 1, 0);
	}
	@Test
	public void testNumNonZero() { 
		for(Tensor tensor : allTensorTypes(10))
			Assert.assertEquals(tensor.setToRandom().add(2).estimateNumNonZeroElements(), 10, 0);
	}
	@Test
	public void testRepeatTensor() {
		Assert.assertEquals(new SparseTensor(10).put(1, 2).add(new RepeatTensor(1, 10)).get(1), 3, 0);
	}
	@Test(expected = IllegalArgumentException.class)
	public void testRepeatTensorNonFinite() {
		new RepeatTensor(Double.NaN, 10);
	}
	@Test(expected = IllegalArgumentException.class)
	public void testRepeatTensorUnderflow() {
		new RepeatTensor(0, 10).get(-1);
	}
	@Test(expected = IllegalArgumentException.class)
	public void testRepeatTensorOverflow() {
		new RepeatTensor(0, 10).get(11);
	}
	@Test(expected = UnsupportedOperationException.class)
	public void testRepeatTensorPut() {
		new RepeatTensor(0, 10).put(1, 0);
	}
	@Test(expected = UnsupportedOperationException.class)
	public void testRepeatTensorCopy() {
		new RepeatTensor(0, 10).copy();
	}
	@Test
	public void testRepeatTensorMultiplication() {
		for(Tensor tensor : allTensorTypes(10)) { 
			new RepeatTensor(1, 10).multiply(tensor);
			tensor.multiply(new RepeatTensor(1, 10));
		}
	}
	@Test(expected = UnsupportedOperationException.class)
	public void testImpossibleMultiplication() {
		new RepeatTensor(1, 10).multiply(new RepeatTensor(2, 10));
	}
}
