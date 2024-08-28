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
	private boolean printOnImproveOnly = false;
	private double bestLoss = Double.POSITIVE_INFINITY;

	/**
	 * Instantiates a {@link VerboseLoss} given one or more comma-separated base
	 * losses to be wrapped. Use a method chain to modify when losses should be
	 * reported, and which output stream is used.
	 * 
	 * @param baseLoss A list of comma-separated {@link Loss} instances.
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
	 * @see #setPrintOnImprovement(boolean)
	 */
	public VerboseLoss setInterval(int every) {
		this.every = every;
		return this;
	}

	/**
	 * Changes by which criteria losses should be printed, that is, on every fixed
	 * count of epochs set by {@link #setInterval(int)} or whenever the primary loss
	 * (the first one enclosed in the constructor) decreases.
	 * 
	 * @param printOnImproveOnly Whether losses should be printed only when the
	 *                           primary loss (which is used for trained parameter
	 *                           selection and early stopping) decreases. Default is
	 *                           false.
	 * @return <code>this</code> verbose loss instance.
	 */
	public VerboseLoss setPrintOnImprovement(boolean printOnImproveOnly) {
		this.printOnImproveOnly = printOnImproveOnly;
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

	/**
	 * Prints the current state of accumulated losses.
	 */
	public void print() {
		String message = "Epoch " + epoch + " ";
		for (int i = 0; i < baseLosses.length; i++)
			message += " " + baseLosses[i].getClass().getSimpleName() + " "
					+ Math.round(Math.abs(values.get(i) / batchCount * 1000)) / 1000.0;
		out.println(message);
	}

	@Override
	public void onEndEpoch() {
		double value = values.get(0);
		if (value < bestLoss)
			bestLoss = value;
		if (printOnImproveOnly) {
			if (value == bestLoss)
				print();
		} else if ((epoch == 0 || epoch % every == 0))
			print();
		values.setToZero();
		batchCount = 0;
		epoch += 1;
	}

	@Override
	public void onEndTraining() {
		epoch = 0;
		bestLoss = Double.POSITIVE_INFINITY;
	}

	@Override
	public double evaluate(Tensor output, Tensor desired) {
		if (values == null)
			values = new DenseTensor(baseLosses.length);
		double value = baseLosses[0].evaluate(output, desired);
		values.putAdd(0, value);
		if (epoch == 0 || epoch % every == 0)
			for (int i = 1; i < baseLosses.length; i++)
				values.putAdd(i, baseLosses[i].evaluate(output, desired));
		batchCount++;
		return value;
	}

	@Override
	public Tensor derivative(Tensor output, Tensor desired) {
		return baseLosses[0].derivative(output, desired);
	}

}
