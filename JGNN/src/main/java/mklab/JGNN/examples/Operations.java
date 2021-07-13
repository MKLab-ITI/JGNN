package mklab.JGNN.examples;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.operations.Add;

public class Operations {

	public static void main(String[] args) throws Exception {
		System.out.println(new Add().run(Tensor.fromDouble(1), Tensor.fromDouble(1)));
	}
}
