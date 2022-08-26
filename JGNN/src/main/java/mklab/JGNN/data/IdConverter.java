package mklab.JGNN.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Slice;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Range;
import mklab.JGNN.nn.operations.Concat;

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
	public Slice getIds() {
		if(idSlice==null)
			idSlice = new Slice(new Range(0, ids.size()));
		return idSlice;
	}
	public Matrix oneHotFromBinary(ArrayList<HashMap<Long, String>> nodeFeatures) {
		IdConverter encoder = new IdConverter();
		Matrix features = new SparseMatrix(size(), nodeFeatures.size());
		for(int feature=0;feature<nodeFeatures.size();feature++)
			for(Entry<Long, String> values : nodeFeatures.get(feature).entrySet()) {
				features.put(values.getKey(), feature, Double.parseDouble(values.getValue()));
			}
		return features;
	}
	public Matrix oneHot(ArrayList<HashMap<Long, String>> nodeFeatures) {
		Matrix features = null;
		for(HashMap<Long, String> feature : nodeFeatures) {
			Matrix onehot = oneHot(feature);
			features = features==null?onehot:(Matrix)new Concat().run(features, onehot);
		}
		return features.setDimensionName(nodeDimensionName, featureDimensionName);
	}
	public Matrix oneHot(HashMap<Long, String> nodeLabels) {
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values())
			if(label!=null)
				encoder.getOrCreateId(label);
		Matrix encoding = new SparseMatrix(size(), encoder.size());
		for(long i=0;i<size();i++)
			if(nodeLabels.get(i)!=null)
				encoding.put(i, encoder.getId(nodeLabels.get(i)), 1);
		return encoding.setDimensionName(nodeDimensionName, featureDimensionName);
	}
	public Matrix oneHot(Tensor nodeLabels) {
		Matrix encoding = new SparseMatrix(size(), 2);
		for(long i=0;i<size();i++)
			encoding.put(i, (long)nodeLabels.get(i), 1);
		return encoding.setDimensionName(nodeDimensionName, featureDimensionName);
	}
	public Matrix oneHot(List<Tensor> nodeLabels) {
		Matrix encoding = new SparseMatrix(size(), nodeLabels.size());
		for(int col=0;col<nodeLabels.size();col++) 
			for(long i=0;i<nodeLabels.get(0).size();i++) 
				encoding.put(i, col, nodeLabels.get(col).get(i));
		return encoding.setDimensionName(nodeDimensionName, featureDimensionName);
	}
	public Tensor toTensor(HashMap<Integer, String> nodeLabels) {
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values()) 
			encoder.getOrCreateId(label);
		if(encoder.size()!=2)
			throw new RuntimeException("Only binary conversion is possible for tensor conversion. Try oneHot encoding enstead.");
		Tensor ret = new DenseTensor(nodeLabels.size());
		for(int i=0;i<size();i++)
			ret.put(i, encoder.getId(nodeLabels.get(i)));
		return ret;
	}
	public List<Tensor> toMultipleTensors(HashMap<Long, String> nodeLabels) {
		List<Tensor> ret = new ArrayList<Tensor>();
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values()) 
			encoder.getOrCreateId(label);
		for(int i=0;i<encoder.size();i++)
			ret.add(new DenseTensor(size()).setDimensionName(nodeDimensionName));
		for(long i=0;i<size();i++)
			ret.get((int)encoder.getId(nodeLabels.get(i))).put(i, 1);
		return ret;
	}
}
