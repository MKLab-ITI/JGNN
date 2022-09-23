package mklab.JGNN.adhoc;

import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.Slice;
import mklab.JGNN.core.util.Range;
/**
 * Converts back-and-forth between objects and unique ids and automates one-hot encoding.
 * @author Emmanouil Krasanakis
 */
public class IdConverter {
	protected HashMap<Object, Long> ids = new HashMap<Object, Long>();
	protected HashMap<Long, Object> inverse = new HashMap<Long, Object>();
	protected String nodeDimensionName;
	protected String featureDimensionName;
	private Slice idSlice = null;
	/**
	 * Instantiates an empty converter to be filled with {@link #getOrCreateId(Object)}.
	 */
	public IdConverter() {
	}
	/**
	 * Instantiates the converter on a list of objects to register
	 * with {@link #getOrCreateId(Object)} on.
	 * @param objects A list of objects.
	 */
	public IdConverter(List<?> objects) {
		for(int i=0;i<objects.size();i++)
			getOrCreateId(objects.get(i));
	}
	/**
	 * Sets dimension names for one-hot encodings.
	 * @param nodeDimensionName The dimension name for traversing nodes (e.g. "node"). 
	 * @param featureDimensionName The dimension name for traversing features (e.g. "label").
	 * @return <code>this</code> instance
	 */
	public IdConverter setDimensionName(String nodeDimensionName, String featureDimensionName) {
		this.nodeDimensionName = nodeDimensionName;
		this.featureDimensionName = featureDimensionName;
		return this;
	}
	/**
	 * Retrieves an identifier for a given object, creating one if none exists.
	 * @param object The object for which to obtain an identifier.
	 * @return A <code>long</code> identifier.
	 * @see #getId(Object)
	 * @see #get(long)
	 */
	public long getOrCreateId(Object object) {
		Long ret = ids.get(object);
		if(ret==null) {
			ids.put(object, ret = (long)ids.size());
			inverse.put(ret, object);
			idSlice = null;
		}
		return ret;
	}
	/**
	 * Retrieves the object corresponding to a given identifier.
	 * @param id The identifier of the object.
	 * @return The object.
	 */
	public Object get(long id) {
		return inverse.get(id);
	}
	/**
	 * Retrieves an identifier.
	 * @param object An object with a registered identifier.
	 * @return A <code>long</code> identifier.
	 * @exception Exception, if the identifiers does not exist.
	 * @see #getOrCreateId(Object)
	 */
	public long getId(Object object) {
		return ids.get(object);
	}
	/**
	 * The number of registered identifiers.
	 * @return A <code>long</code> value.
	 */
	public long size() {
		return ids.size();
	}
	/**
	 * Checks whether the object has been registered with {@link #getOrCreateId(Object)}.
	 * @param object An object to check if it exists.
	 * @return A boolean value.
	 */
	public boolean contains(Object object) {
		return ids.containsKey(object);
	}
	/**
	 * Returns a slice of all registered identifiers.
	 * The slice is persistent across multiple calls to this method, but is 
	 * instantiated anew after {@link #getOrCreateId(Object)} registers a new
	 * object (but not if it retrieves an existing object). 
	 * 
	 * @return A {@link Slice}.
	 */
	public Slice getSlice() {
		if(idSlice==null)
			idSlice = new Slice(new Range(0, ids.size()));
		return idSlice;
	}
}
