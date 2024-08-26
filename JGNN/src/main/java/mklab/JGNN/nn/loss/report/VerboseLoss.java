package mklab.JGNN.nn.loss.report;

import java.io.PrintStream;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements a {@link Loss} that wraps other losses and outputs their value
 * during training to an output stream (to {@link System#out} by default). This
 * is the simplest loss wrapper to keep track of training progress.
 * 
 * @author Emmanouil Krasanakis
 * @see VerboseLoss#VerboseLoss(Loss)
 */
public class VerboseLoss extends Loss {
	private int epoch = 0;
	private int every = 1;
	private Loss[] baseLosses;
	private PrintStream out;
	private Tensor values;
	private int batchCount = 0;

	/**
	 * Instantiates a {@link VerboseLoss} given one or more comma-separated base losses 
	 * to be wrapped. Use a method chain to modify when losses should be reported, and which output
	 * stream is used.
	 * 
	 * @param baseLoss
	 * @see #setInterval(int)
	 * @see #setStream(PrintStream)
	 */
	public VerboseLoss(Loss... baseLosses) {
		this.baseLosses = baseLosses;
		out = System.out;
	}

	/**
	 * Changes on which epochs the loss should be reported.
	 * 
	 * @param every The loss is reported on epochs 0, every, 2every, ... Default is
	 *              1.
	 * @return <code>this</code> verbose loss instance.
	 */
	public VerboseLoss setInterval(int every) {
		this.every = every;
		return this;
	}

	/**
	 * Changes where the output is printed.
	 * 
	 * @param out The print stream to print to. Default is {@link System#out}.
	 * @return <code>this</code> verbose loss instance.
	 */
	public VerboseLoss setStream(PrintStream out) {
		this.out = out;
		return this;
	}
	
	
	public void print() {
		String message = "Epoch " + epoch + " ";
		for(int i=0;i<baseLosses.length;i++) 
			message += " " + baseLosses[i].getClass().getSimpleName() + " " + Math.round(Math.abs(values.get(i)/batchCount * 1000)) / 1000.0;
		out.println(message);
	}
	
	@Override
	public void onEndEpoch() {
		if (epoch == 0 || epoch % every == 0) 
			print();
		values.setToZero();
		batchCount = 0;
		epoch += 1;
	}
	
	@Override
	public void onEndTraining() {
		epoch = 0;
	}

	@Override
	public double evaluate(Tensor output, Tensor desired) {
		if(values==null)
			values = new DenseTensor(baseLosses.length);
		double value = baseLosses[0].evaluate(output, desired);
		values.putAdd(0, value);
		if (epoch == 0 || epoch % every == 0) 
			for(int i=1;i<baseLosses.length;i++)
				values.putAdd(i, baseLosses[i].evaluate(output, desired));
		batchCount++;
		return value;
	}

	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		return baseLosses[0].derivative(output, desired);
	}

}
