package mklab.JGNN.examples;

import mklab.JGNN.core.Model;
import mklab.JGNN.core.NNOperation;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.inputs.Constant;
import mklab.JGNN.core.inputs.Variable;
import mklab.JGNN.core.operations.Add;
import mklab.JGNN.core.operations.Log;
import mklab.JGNN.core.operations.Multiply;

public class ModelCustomDefinition {

	public static void main(String[] args) {
		Variable x = new Variable();
		Constant c1 = new Constant(Tensor.fromDouble(1));
		Constant c2 = new Constant(Tensor.fromDouble(2));
		NNOperation mult = new Multiply().addInput(x).addInput(c2);
		NNOperation add = new Add().addInput(mult).addInput(c1);
		NNOperation log = new Log().addInput(add);
		
		Model model = new Model().addInput(x).addOutput(log);
		System.out.println(model.predict(Tensor.fromDouble(2)));
		System.out.println(add.getPrediction());
	}

}
