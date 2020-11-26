package mklab.JGNN.core.operations;

import mklab.JGNN.core.primitives.Matrix;
import mklab.JGNN.core.primitives.Optimizer;
import mklab.JGNN.core.primitives.Tensor;
import mklab.JGNN.core.primitives.matrix.DenseMatrix;
import mklab.JGNN.core.primitives.tensor.DenseTensor;
import mklab.JGNN.core.util.Loss;

public class LSTM {
	private Matrix ui, wi, uf, wf, uo, wo, ug, wg;
	private Matrix tape_ui, tape_wi, tape_uf, tape_wf, tape_uo, tape_wo, tape_ug, tape_wg;
	private Optimizer optimizer;
	
	public static class LSTMState {
		private Tensor previousMemory;
		private Tensor previousOutput;
		public LSTMState(Tensor previousMemory, Tensor previousOutput) {
			this.previousMemory = previousMemory;
			this.previousOutput = previousOutput;
		}
		public Tensor getMemory() {
			return previousMemory;
		}
		public Tensor getOutput() {
			return previousOutput;
		}
	}
	
	public LSTM(Optimizer optimizer, int inputSize, int outputSize) {
		this.optimizer = optimizer;
		int memorySize = outputSize;
		ui = new DenseMatrix(memorySize, inputSize);
		uf = new DenseMatrix(memorySize, inputSize);
		uo = new DenseMatrix(memorySize, inputSize);
		ug = new DenseMatrix(memorySize, inputSize);
		wi = new DenseMatrix(memorySize, outputSize);
		wf = new DenseMatrix(memorySize, outputSize);
		wo = new DenseMatrix(memorySize, outputSize);
		wg = new DenseMatrix(memorySize, outputSize);
		ui.setToRandom();
		uf.setToRandom();
		uo.setToRandom();
		ug.setToRandom();
		wi.setToRandom();
		wf.setToRandom();
		wo.setToRandom();
		wg.setToRandom();
	}
	
	protected LSTM() {
	}
	
	public Optimizer getOptimizer() {
		return optimizer;
	}
	
	public LSTMState createFirstState() {
		return new LSTMState(new DenseTensor(ui.getRows()), new DenseTensor(wi.getCols()));
	}
	
	public LSTMState output(Tensor input, LSTMState previousState) {
		Tensor previousMemory = previousState.getMemory();
		Tensor previousOutput = previousState.getOutput();
		Tensor i = Loss.sigmoid(ui.transform(input).selfAdd(wi.transform(previousOutput)));
		Tensor f = Loss.sigmoid(uf.transform(input).selfAdd(wf.transform(previousOutput)));
		Tensor o = Loss.sigmoid(uo.transform(input).selfAdd(wo.transform(previousOutput)));
		Tensor memoryGate = Loss.tanh(ug.transform(input).selfAdd(wg.transform(previousOutput)));
		Tensor memory = Loss.sigmoid(f.selfMultiply(previousMemory).selfAdd(i.selfMultiply(memoryGate)));
		Tensor output = Loss.tanh(memory).selfMultiply(o);

		/*System.out.println("------------ "+this);
		System.out.println("Input "+input.describe());
		System.out.println("Prev memory "+previousMemory.describe());
		System.out.println("Prev output "+previousOutput.describe());
		System.out.println("i "+i.describe());
		System.out.println("f "+f.describe());
		System.out.println("o "+o.describe());
		System.out.println("memoryGate "+memoryGate.describe());
		System.out.println("memory "+memory.describe());
		System.out.println("output "+output.describe());*/
		
		return new LSTMState(memory, output);
	}
	
	public void startTape() {
		tape_ui = ui.zeroCopy();
		tape_uf = uf.zeroCopy();
		tape_uo = uo.zeroCopy();
		tape_ug = ug.zeroCopy();
		tape_wi = wi.zeroCopy();
		tape_wf = wf.zeroCopy();
		tape_wo = wo.zeroCopy();
		tape_wg = wg.zeroCopy();
	}
	
	public double train(Tensor[] inputs, Tensor output) {
		LSTMState[] states = new LSTMState[inputs.length+1];
		states[0] = createFirstState();
		for(int i=0;i<inputs.length;i++)
			states[i+1] = output(inputs[i], states[i]);
		Tensor error = states[inputs.length].getOutput().subtract(output);
		Tensor topError = error;
		for(int i=inputs.length-1;i>=0;i--) 
			error = updateTape(inputs[i], states[i], error);
		return topError.norm();
	}
	
	public void trainOnOutputError(Tensor[] inputs, Tensor outputGradient) {
		LSTMState[] states = new LSTMState[inputs.length+1];
		states[0] = createFirstState();
		for(int i=0;i<inputs.length;i++)
			states[i+1] = output(inputs[i], states[i]);
		Tensor error = outputGradient;
		for(int i=inputs.length-1;i>=0;i--) 
			error = updateTape(inputs[i], states[i], error);
	}
	
	public Tensor predict(Tensor[] inputs) {
		LSTMState state = createFirstState();
		for(int i=0;i<inputs.length;i++)
			state = output(inputs[i], state);
		return state.getOutput();
	}
	
	public Tensor updateTape(Tensor input, LSTMState previousState, Tensor outputErrorGradient) {
		Tensor previousMemory = previousState.getMemory();
		Tensor previousOutput = previousState.getOutput();
		Tensor i = Loss.sigmoid(ui.transform(input).selfAdd(wi.transform(previousOutput)));
		Tensor f = Loss.sigmoid(uf.transform(input).selfAdd(wf.transform(previousOutput)));
		Tensor o = Loss.sigmoid(uo.transform(input).selfAdd(wo.transform(previousOutput)));
		Tensor memoryGate = Loss.tanh(ug.transform(input).selfAdd(wg.transform(previousOutput)));
		Tensor memory = Loss.sigmoid(f.multiply(previousMemory).selfAdd(i.multiply(memoryGate)));
		//Tensor output = Loss.tanh(memory).selfMultiply(o);
		
		Tensor gradient_o = Loss.tanh(memory).selfMultiply(outputErrorGradient);
		Tensor gradient_memory = Loss.tanhDerivative(memory).selfMultiply(o).selfMultiply(outputErrorGradient);
		Tensor gradient_memory_tradeoff = Loss.sigmoidDerivative(f.multiply(previousMemory).selfAdd(i.multiply(memoryGate))).selfMultiply(gradient_memory);
		
		Tensor gradient_f = gradient_memory_tradeoff.multiply(previousMemory);
		Tensor gradient_i = gradient_memory_tradeoff.multiply(memoryGate);
		Tensor gradient_memoryGate = gradient_memory_tradeoff.multiply(i);
		
		Tensor gradient_memoryGate_tradeoff = Loss.tanhDerivative(ug.transform(input).selfAdd(wg.transform(previousOutput))).selfMultiply(gradient_memoryGate);
		Matrix gradient_ug = Matrix.external(gradient_memoryGate_tradeoff, input);
		Matrix gradient_wg = Matrix.external(gradient_memoryGate_tradeoff, previousOutput);
		
		Tensor gradient_o_tradeoff = Loss.sigmoidDerivative(uo.transform(input).selfAdd(wo.transform(previousOutput))).selfMultiply(gradient_o);
		Matrix gradient_uo = Matrix.external(gradient_o_tradeoff, input);
		Matrix gradient_wo = Matrix.external(gradient_o_tradeoff, previousOutput);
		
		Tensor gradient_f_tradeoff = Loss.sigmoidDerivative(uf.transform(input).selfAdd(wf.transform(previousOutput))).selfMultiply(gradient_f);
		Matrix gradient_uf = Matrix.external(gradient_f_tradeoff, input);
		Matrix gradient_wf = Matrix.external(gradient_f_tradeoff, previousOutput);
		
		Tensor gradient_i_tradeoff = Loss.sigmoidDerivative(ui.transform(input).selfAdd(wi.transform(previousOutput))).selfMultiply(gradient_i);
		Matrix gradient_ui = Matrix.external(gradient_i_tradeoff, input);
		Matrix gradient_wi = Matrix.external(gradient_i_tradeoff, previousOutput);
		
		/*System.out.println("Gradient ui "+gradient_ui);
		System.out.println("Gradient gradient_i_tradeoff "+gradient_i_tradeoff);
		System.out.println("Gradient gradient_i "+gradient_i);
		System.out.println("Gradient gradient_memory_tradeoff "+gradient_memory_tradeoff);
		System.out.println("Gradient gradient_memory "+gradient_memory);
		System.out.println("o "+o);
		System.out.println("Loss.tanhDerivative(memory) "+Loss.tanhDerivative(memory));
		System.out.println("outputErrorGradient "+outputErrorGradient);*/
		
		tape_ui.selfAdd(gradient_ui);
		tape_uf.selfAdd(gradient_uf);
		tape_uo.selfAdd(gradient_uo);
		tape_ug.selfAdd(gradient_ug);
		tape_wi.selfAdd(gradient_wi);
		tape_wf.selfAdd(gradient_wf);
		tape_wo.selfAdd(gradient_wo);
		tape_wg.selfAdd(gradient_wg);
		
		Tensor gradient_previousOutput = wg.transposed().transform(gradient_memoryGate_tradeoff)
				.selfAdd(wi.transposed().transform(gradient_i_tradeoff))
				.selfAdd(wf.transposed().transform(gradient_f_tradeoff))
				.selfAdd(wo.transposed().transform(gradient_o_tradeoff));
		
		return gradient_previousOutput;
	}
	
	public void endTape() {
		optimizer.update(ui, tape_ui);
		optimizer.update(uf, tape_uf);
		optimizer.update(uo, tape_uo);
		optimizer.update(ug, tape_ug);
		optimizer.update(wi, tape_wi);
		optimizer.update(wf, tape_wf);
		optimizer.update(wo, tape_wo);
		optimizer.update(wg, tape_wg);
		tape_ui = null;
		tape_uf = null;
		tape_uo = null;
		tape_ug = null;
		tape_wi = null;
		tape_wf = null;
		tape_wo = null;
		tape_wg = null;
	}

	public void aggregate(LSTM lstm) {
		ui.selfAdd(lstm.ui).selfMultiply(0.5);
		uf.selfAdd(lstm.uf).selfMultiply(0.5);
		uo.selfAdd(lstm.uo).selfMultiply(0.5);
		ug.selfAdd(lstm.ug).selfMultiply(0.5);
		wi.selfAdd(lstm.ui).selfMultiply(0.5);
		wf.selfAdd(lstm.uf).selfMultiply(0.5);
		wo.selfAdd(lstm.uo).selfMultiply(0.5);
		wg.selfAdd(lstm.ug).selfMultiply(0.5);
	}
}
