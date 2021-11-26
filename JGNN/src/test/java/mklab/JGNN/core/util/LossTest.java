package mklab.JGNN.core.util;

import org.junit.Test;
import org.junit.Assert;

public class LossTest {
	@Test
	public void testFiniteLoss() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropy(1, 1)));
	}

	@Test
	public void testFiniteDerivative() {
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 0)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(0, 1)));
		Assert.assertTrue(Double.isFinite(Loss.crossEntropyDerivative(1, 1)));
	}

	@Test(expected = IllegalArgumentException.class)
	public void testNonbinaryCrossEntropyDerivativeLabel() {
		Loss.crossEntropyDerivative(0, 0.1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testNegativeCrossEntropyDerivativeLabel() {
		Loss.crossEntropyDerivative(0, -1);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testNegativeCrossEntropyDerivativePrediction() {
		Loss.crossEntropyDerivative(-1, 0);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testLargeCrossEntropyDerivativeLabel() {
		Loss.crossEntropyDerivative(2, 0);
	}
}
