package mklab.JGNN;

import org.junit.Assert;
import org.junit.Test;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;

public class TensorTest {
	@Test
	public void testNumeric() {
		Tensor tensor1 = new DenseTensor(1, 2, 0);
		Tensor tensor2 = new DenseTensor(4, 5, 3);
		Assert.assertTrue(tensor1.get(1)==2);
		Assert.assertTrue(tensor1.add(tensor2).get(1)==7);
		Assert.assertTrue(tensor1.subtract(tensor2).get(1)==-3);
		Assert.assertTrue(tensor1.multiply(tensor2).get(1)==10);
		Assert.assertTrue(tensor1.inverse().get(2) == 0);
		Assert.assertTrue(tensor1.inverse().get(1) == 0.5);
		Assert.assertTrue(tensor1.negative().get(1) == -2);
	}
}
