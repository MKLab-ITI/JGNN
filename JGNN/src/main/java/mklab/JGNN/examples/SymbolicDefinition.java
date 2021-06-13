package mklab.JGNN.examples;

import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Tensor;

public class SymbolicDefinition {

	public static void main(String[] args) {
		ModelBuilder modelBuilder = new ModelBuilder()
				.var("x") // first argument
				.constant("a", Tensor.fromDouble(2))
				.constant("b", Tensor.fromDouble(1))
				.operation("yhat = a*x+b")
				.out("yhat");
		System.out.println(modelBuilder.getModel().predict(Tensor.fromDouble(2)));
	}

}
