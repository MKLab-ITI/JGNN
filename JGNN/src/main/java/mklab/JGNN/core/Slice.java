package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.tensor.DenseTensor;

/**
 * This class provices an interface with which to define data slices, for
 * instance to sample labels.
 * 
 * @author Emmanouil Krasanakis
 */
public class Slice implements Iterable<Long> {
	private List<Long> ids;

	/**
	 * Instantiates a data slice from a collection of element identifiers.
	 * 
	 * @param collection An iterable of longs.
	 */
	public Slice(Iterable<Long> collection) {
		this.ids = new ArrayList<Long>();
		for (long id : collection)
			this.ids.add(id);
	}

	/**
	 * Shuffles the slice.
	 * 
	 * @return <code>this</code> slice.
	 * @see #shuffle(int)
	 */
	public Slice shuffle() {
		Collections.shuffle(ids);
		return this;
	}

	/**
	 * Shuffles the slice with a provided randomization seed.
	 * 
	 * @return <code>this</code> slice.
	 * @param seed The seed to shuffle with.
	 * @return <code>this</code> slice.
	 * @see #shuffle()
	 */
	public Slice shuffle(int seed) {
		Collections.shuffle(ids, new Random(seed));
		return this;
	}

	/**
	 * Obtains the identifiers in a given range of the (shuffled) slice.
	 * 
	 * @param from The beginning of the identifiers' position in the slice.
	 * @param end  The end (non-inclusive) of the identifiers' position in the
	 *             slice.
	 * @return A new Slice instance holding the position identifiers in this one's
	 *         given range.
	 * 
	 * @see #range(double, double)
	 */
	public Slice range(int from, int end) {
		return new Slice(ids.subList(from, end));
	}

	/**
	 * Constructs a column matrix holding identifiers in the range
	 * 0,1,..{@link #size()}-1 so that the pattern
	 * <code>slice.samplesAsFeatures().accessRows(slice.range(from, end))</code>
	 * retrieves one-element tensors holding
	 * <code>slice[from], slice[from+1], ... slice[end]</code>. The constructed
	 * matrix is typically used as node identifier data.
	 * 
	 * This is different than {@link #asTensor()}.
	 * 
	 * @return A {@link Matrix}.
	 */
	public Matrix samplesAsFeatures() {
		return Tensor.fromRange(0, size()).asColumn();
	}

	/**
	 * Performs the {@link #range(int, int)} operation while replacing values of
	 * <code>from</code> and <code>end</code> with <code>(int)(from*size())</code>
	 * and <code>(int)(end*size())</code> so that fractional ranges can be obtained.
	 * For example, you can call <code>slice.shuffle().range(0.5, 1)</code> to
	 * obtain a random subset of the slice's identifiers.
	 * 
	 * @param from An integer at least 1 or a double in the range [0,1).
	 * @param end  An integer greater than 1 or a double in the range [0,1].
	 * @return A new Slice instance holding the position identifiers in this one's
	 *         given range.
	 * @see #size()
	 */
	public Slice range(double from, double end) {
		if (from < 1)
			from = (int) (from * size());
		if (end <= 1)
			end = (int) (end * size());
		return range((int) from, (int) end);
	}

	/**
	 * Retrieves the size of the slice.
	 * 
	 * @return An integer.
	 */
	public int size() {
		return ids.size();
	}

	@Override
	public Iterator<Long> iterator() {
		return ids.iterator();
	}

	/**
	 * Creates a dense tensor holding the slice's identifiers.
	 * 
	 * @return A {@link DenseTensor}.
	 */
	public Tensor asTensor() {
		return new DenseTensor(iterator());
	}

}
