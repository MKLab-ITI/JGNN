package mklab.JGNN.core.tensor;

import java.util.ArrayList;
import java.util.Iterator;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.util.Range;

/**
 * This class provides a dense {@link Tensor} that wraps an array of doubles.
 * 
 * @author Emmanouil Krasanakis
 */
public class VectorizedTensor extends Tensor {
    public double[] values;
    public static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Constructs a dense tensor from an iterator holding that outputs its values.
     * Tensor size is equal to the number of extracted values.
     * 
     * @param iterator The iterator to obtain values from.
     */
    public VectorizedTensor(Iterator<? extends Number> iterator) {
        ArrayList<Number> list = new ArrayList<Number>();
        iterator.forEachRemaining(list::add);
        init(list.size());
        for (int i = 0; i < list.size(); ++i)
            put(i, list.get(i).doubleValue());
    }

    public VectorizedTensor(double... values) {
        this(values.length);
        int i = 0;
        int bound = SPECIES.loopBound(values.length);
        for (; i < bound; i += SPECIES.length()) {
            DoubleVector vec = DoubleVector.fromArray(SPECIES, values, i);
            vec.intoArray(this.values, i);
        }
        for (; i < values.length; ++i) {
            this.values[i] = values[i];
        }
    }

    /**
     * Constructs a dense tensor holding zero values.
     * 
     * @param size The size of the tensor.
     */
    public VectorizedTensor(long size) {
        super(size);
    }

    /**
     * Reconstructs a serialized Tensor (i.e. the outcome of {@link #toString()})
     * 
     * @param expr A serialized tensor
     * @throws IllegalArgumentException If the serialization is null or empty.
     */
    public VectorizedTensor(String expr) {
        if (expr == null)
            throw new IllegalArgumentException("Cannot create tensor from a null string");
        if (expr.length() == 0) {
            init(0);
            return;
        }
        String[] splt = expr.split(",");
        init(splt.length);
        for (int i = 0; i < splt.length; ++i)
            put(i, Double.parseDouble(splt[i]));
    }

    public VectorizedTensor() {
        this(0);
    }

    public final Tensor put(long pos, double value) {
        values[(int) pos] = value;
        return this;
    }
    
    public final Tensor put(int pos, double value) {
        values[pos] = value;
        return this;
    }
    
    public final void putAdd(int pos, double value) {
        values[pos] += value;
    }

    public final double get(long pos) {
        return values[(int) pos];
    }
    
    public final double get(int pos) {
        return values[pos];
    }

    @Override
    protected void allocate(long size) {
        values = new double[(int) size];
    }

    @Override
    public Tensor zeroCopy(long size) {
    	if(size<100000)
    		return new DenseTensor(size);
        return new VectorizedTensor(size);
    }

    @Override
    public Iterator<Long> traverseNonZeroElements() {
        return new Range(0, size());
    }

    @Override
    public void release() {
        values = null;
    }

    @Override
    public void persist() {
    }

    @Override
    public Tensor add(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;
            VectorizedTensor res = (VectorizedTensor) zeroCopy();

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.add(vec2).intoArray(res.values, i);
            }
            for (; i < size(); ++i) 
                res.values[i] = values[i] + other.values[i];
            return res;
        }
        return super.add(tensor);
    }

    @Override
    public Tensor selfAdd(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.add(vec2).intoArray(values, i);
            }
            for (; i < size(); ++i) 
                values[i] += other.values[i];

            return this;
        } 
        return super.selfAdd(tensor);
    }

    @Override
    public Tensor subtract(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;
            VectorizedTensor res = (VectorizedTensor) zeroCopy();

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.sub(vec2).intoArray(res.values, i);
            }
            for (; i < size(); ++i) 
                res.values[i] = values[i] - other.values[i];
            return res;
        }
        return super.subtract(tensor);
    }

    @Override
    public Tensor selfSubtract(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.sub(vec2).intoArray(values, i);
            }
            for (; i < size(); ++i) 
                values[i] -= other.values[i];
            return this;
        }
        return super.selfSubtract(tensor);
    }

    @Override
    public Tensor multiply(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;
            VectorizedTensor res = (VectorizedTensor) zeroCopy();

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.mul(vec2).intoArray(res.values, i);
            }
            for (; i < size(); ++i) 
                res.values[i] = values[i] * other.values[i];
            return res;
        } 
        return super.multiply(tensor);
    }

    @Override
    public Tensor selfMultiply(Tensor tensor) {
        if (tensor instanceof VectorizedTensor) {
            VectorizedTensor other = (VectorizedTensor) tensor;

            int i = 0;
            int bound = SPECIES.loopBound((int) size());
            for (; i < bound; i += SPECIES.length()) {
                DoubleVector vec1 = DoubleVector.fromArray(SPECIES, values, i);
                DoubleVector vec2 = DoubleVector.fromArray(SPECIES, other.values, i);
                vec1.mul(vec2).intoArray(values, i);
            }
            for (; i < size(); ++i) 
                values[i] *= other.values[i];
            return this;
        }
        return super.selfMultiply(tensor);
    }

    @Override
    public Tensor multiply(double value) {
        VectorizedTensor res = (VectorizedTensor) zeroCopy();

        int i = 0;
        int bound = SPECIES.loopBound((int) size());
        DoubleVector valueVector = DoubleVector.broadcast(SPECIES, value);
        for (; i < bound; i += SPECIES.length()) {
            DoubleVector vec = DoubleVector.fromArray(SPECIES, values, i);
            vec.mul(valueVector).intoArray(res.values, i);
        }
        for (; i < size(); ++i) 
            res.values[i] = values[i] * value;
        return res;
    }

    @Override
    public Tensor selfMultiply(double value) {
        int i = 0;
        int bound = SPECIES.loopBound((int) size());
        DoubleVector valueVector = DoubleVector.broadcast(SPECIES, value);
        for (; i < bound; i += SPECIES.length()) {
            DoubleVector vec = DoubleVector.fromArray(SPECIES, values, i);
            vec.mul(valueVector).intoArray(values, i);
        }
        for (; i < size(); ++i) 
            values[i] *= value;
        return this;
    }
}
