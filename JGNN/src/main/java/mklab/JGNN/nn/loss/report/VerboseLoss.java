package mklab.JGNN.nn.loss.report;

import java.io.PrintStream;

import mklab.JGNN.core.Tensor;
import mklab.JGNN.nn.Loss;

/**
 * Implements a {@link Loss} that wraps other losses and outputs their value during training to an output stream
 * (to {@link System#out} by default). This is the simplest loss wrapper to keep track of training progress.
 * 
 * @author Emmanouil Krasanakis
 * @see VerboseLoss#VerboseLoss(Loss)
 */
public class VerboseLoss extends Loss {
	private int epoch = 0;
	private int every = 1;
	private Loss baseLoss;
	private PrintStream out;
	
	public void reset() {
		epoch = 0;
	}
	
	/**
	 * Instantiates a {@link VerboseLoss} given a base loss to be wrapped.
	 * Use a method chain to modify when losses should be reported, and which
	 * output stream is used.
	 * @param baseLoss
	 * @see #setInterval(int)
	 * @see #setStream(PrintStream)
	 */
	public VerboseLoss(Loss baseLoss) {
		this.baseLoss = baseLoss;
		out = System.out;
	}
	
	/**
	 * Changes on which epochs the loss should be reported.
	 * @param every The loss is reported on epochs 0, every, 2every, ... Default is 1.
	 * @return <code>this</code> verbose loss instance.
	 */
	public VerboseLoss setInterval(int every) {
		this.every = every;
		return this;
	}
	
	/**
	 * Changes where the output is printed.
	 * @param out The print stream to print to. Default is {@link System#out}.
	 * @return <code>this</code> verbose loss instance.
	 */
	public VerboseLoss setStream(PrintStream out) {
		this.out = out;
		return this;
	}
	
	@Override
	public double evaluate(Tensor output, Tensor desired) {
		epoch += 1;
		double value = baseLoss.evaluate(output, desired);
		if(epoch==0 || epoch%every==0)
			out.println("Epoch "+epoch+" "+baseLoss.getClass().getSimpleName()+" "+Math.round(Math.abs(value*1000))/1000.0);
		return value;
	}

	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		return baseLoss.derivative(output, desired);
	}

}
