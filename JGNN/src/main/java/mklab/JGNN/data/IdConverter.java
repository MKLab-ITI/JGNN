package mklab.JGNN.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;

public class IdConverter {
	protected HashMap<Object, Integer> ids = new HashMap<Object, Integer>();
	public IdConverter() {
	}
	public IdConverter(ArrayList<?> objects) {
		for(int i=0;i<objects.size();i++)
			getOrCreateId(objects.get(i));
	}
	public int getOrCreateId(Object object) {
		Integer ret = ids.get(object);
		if(ret==null)
			ids.put(object, ret = ids.size());
		return ret;
	}
	public int getId(Object object) {
		return ids.get(object);
	}
	public int size() {
		return ids.size();
	}
	public boolean contains(Object object) {
		return ids.containsKey(object);
	}

	public Matrix oneHot(HashMap<Integer, String> nodeLabels) {
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values())
			encoder.getOrCreateId(label);
		Matrix encoding = new SparseMatrix(nodeLabels.size(), encoder.size());
		for(int i=0;i<size();i++)
			encoding.put(i, encoder.getId(nodeLabels.get(i)), 1);
		return encoding;
	}
	public Matrix oneHot(Tensor nodeLabels) {
		Matrix encoding = new SparseMatrix(nodeLabels.size(), 2);
		for(long i=0;i<size();i++)
			encoding.put(i, (long)nodeLabels.get(i), 1);
		return encoding;
	}
	public Matrix oneHot(List<Tensor> nodeLabels) {
		Matrix encoding = new SparseMatrix(size(), nodeLabels.size());
		for(int col=0;col<nodeLabels.size();col++) 
			for(long i=0;i<nodeLabels.get(0).size();i++) 
				encoding.put(i, col, nodeLabels.get(col).get(i));
		return encoding;
	}
	public Tensor toTensor(HashMap<Integer, String> nodeLabels) {
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values()) 
			encoder.getOrCreateId(label);
		if(encoder.size()!=2)
			throw new RuntimeException("Only binary conversion is possible for tensor conversion. Try oneHot encoding enstead");
		Tensor ret = new DenseTensor(nodeLabels.size());
		for(int i=0;i<size();i++)
			ret.put(i, encoder.getId(nodeLabels.get(i)));
		return ret;
	}
	public List<Tensor> toMultipleTensors(HashMap<Integer, String> nodeLabels) {
		List<Tensor> ret = new ArrayList<Tensor>();
		IdConverter encoder = new IdConverter();
		for(String label : nodeLabels.values()) 
			encoder.getOrCreateId(label);
		for(int i=0;i<encoder.size();i++)
			ret.add(new DenseTensor(size()));
		for(int i=0;i<size();i++)
			ret.get(encoder.getId(nodeLabels.get(i))).put(i, 1);
		return ret;
	}
}
