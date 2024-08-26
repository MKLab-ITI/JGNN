package mklab.JGNN.nn.optimizers;

import java.util.HashMap;

import mklab.JGNN.nn.Optimizer;
import mklab.JGNN.core.Tensor;

/**
 * Thic class implements an Adam {@link Optimizer} as explained in the paper:
 * <i>Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic
 * optimization." arXiv preprint arXiv:1412.6980 (2014). </i> <br>
 * It also supports the NDAdam improvement, which ports advantages of SGD to
 * Adam, as introduced in the paper: <i>Zhang, Zijun. "Improved adam optimizer
 * for deep neural networks." 2018 IEEE/ACM 26th International Symposium on
 * Quality of Service (IWQoS). IEEE, 2018. </i>
 * 
 * @author Emmanouil Krasanakis
 */
public class Adam implements Optimizer {
	private double b1;
	private double b2;
	private double learningRate;
	private double espilon;
	private boolean NDmode;

	private HashMap<Tensor, Tensor> m = new HashMap<Tensor, Tensor>();
	private HashMap<Tensor, Tensor> v = new HashMap<Tensor, Tensor>();
	private HashMap<Tensor, Double> b1t = new HashMap<Tensor, Double>();
	private HashMap<Tensor, Double> b2t = new HashMap<Tensor, Double>();

	/**
	 * Initializes an NDAdam instance of an {@link Adam} optimizer with the default
	 * parameters recommended by the papers.
	 */
	public Adam() {
		this(false, 0.001);
	}

	/**
	 * Initializes an NDAdam instance of an {@link Adam} optimizer with the default
	 * parameters recommended by the papers but allows for the specification of the
	 * learning rate.
	 * 
	 * @param learningRate The learning rate.
	 */
	public Adam(double learningRate) {
		this(false, learningRate, 0.9, 0.999);
	}

	/**
	 * Initializes an {@link Adam} optimizer with the default parameters recommended
	 * in the literature, but allows for the specification of the learning rate and
	 * whether NDAdam or simple Adam is used.
	 * 
	 * @param NDmode       Should be true to use NDAdam and false to use simple Adam
	 *                     optimization.
	 * @param learningRate The learning rate.
	 */
	public Adam(boolean NDmode, double learningRate) {
		this(NDmode, learningRate, 0.9, 0.999);
	}

	/**
	 * Initializes an instance of an {@link Adam} optimizer with the default
	 * parameters while customizing the variation and learning rate.
	 * 
	 * @param NDmode       Should be true to use NDAdam and false to use simple Adam
	 *                     optimization.
	 * @param learningRate The learning rate.
	 * @param b1           Adam's b1 parameter.
	 * @param b2           Adam's b2 parameter.
	 */
	public Adam(boolean NDmode, double learningRate, double b1, double b2) {
		this(NDmode, learningRate, b1, b2, 1.E-8);
	}

	/**
	 * Initializes an {@link Adam} optimizer by customizing all arguments.
	 * 
	 * @param NDmode       Should be true to use NDAdam and false to use simple Adam
	 *                     optimization.
	 * @param learningRate The learning rate.
	 * @param b1           Adam's b1 parameter.
	 * @param b2           Adam's b2 parameter.
	 * @param epsilon      Adam's numerical tolerance.
	 */
	public Adam(boolean NDmode, double learningRate, double b1, double b2, double epsilon) {
		if (b1 < 0 || b1 >= 1)
			throw new IllegalArgumentException("b1 values for Adam should be in the range [0,1) but given " + b1);
		if (b2 < 0 || b2 >= 1)
			throw new IllegalArgumentException("b2 values for Adam should be in the range [0,1) but given " + b2);
		if (epsilon <= 0 || epsilon >= 1)
			throw new IllegalArgumentException(
					"epsilon values for Adam should be in the range (0,1) but given " + epsilon);
		this.NDmode = NDmode;
		this.learningRate = learningRate;
		this.b1 = b1;
		this.b2 = b2;
		this.espilon = epsilon;
	}

	@Override
	public void update(Tensor value, Tensor gradient) {
		synchronized (value) {
			Tensor mValue = m.get(value);
			Tensor vValue = v.get(value);
			if (mValue == null) {
				m.put(value, mValue = value.zeroCopy());
				v.put(value, vValue = value.zeroCopy());
			}
			// Tensor val = value.copy().setToNormalized();
			if (NDmode)
				gradient = gradient.subtract(value.multiply(gradient.dot(value)));
			b1t.put(value, b1t.getOrDefault(value, 1.) * b1);
			b2t.put(value, b2t.getOrDefault(value, 1.) * b2);

			mValue.selfMultiply(b1).selfAdd(gradient.multiply(1 - b1));
			vValue.selfMultiply(b2).selfAdd(gradient.multiply(gradient).selfMultiply(1 - b2));

			Tensor mHat = mValue.multiply(1. / (1 - b1t.get(value)));
			Tensor vHat = vValue.multiply(1. / (1 - b2t.get(value)));
			value.selfAdd(
					mHat.selfMultiply(-learningRate).selfMultiply(vHat.selfAdd(espilon).selfSqrt().selfInverse()));
			if (NDmode)
				value.setToNormalized();
		}
	}

	@Override
	public void reset() {
		m = new HashMap<Tensor, Tensor>();
		v = new HashMap<Tensor, Tensor>();
		b1t = new HashMap<Tensor, Double>();
		b2t = new HashMap<Tensor, Double>();
	}
}
