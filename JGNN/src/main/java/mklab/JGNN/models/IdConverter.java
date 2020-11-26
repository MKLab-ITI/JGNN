package mklab.JGNN.models;

import java.util.ArrayList;
import java.util.HashMap;

public class IdConverter {
	protected HashMap<Object, Integer> ids = new HashMap<Object, Integer>();
	public IdConverter(ArrayList<?> objects) {
		for(int i=0;i<objects.size();i++)
			ids.put(objects.get(i), i);
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
}
