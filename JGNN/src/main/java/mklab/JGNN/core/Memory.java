package mklab.JGNN.core;

import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Stack;
import java.util.WeakHashMap;

/**
 * A memory management systems for thread-safe allocation and release of arrays of doubles.
 * Soft references to allocated arrays kept so that released ones can be reused by future
 * allocation calls without explicitly initializing memory.
 * @author Emmanouil Krasanakis
 */
public class Memory {
	public static class Scope {
		private Stack<ArrayList<WeakReference<double[]>>> values = new Stack<ArrayList<WeakReference<double[]>>>();
		private ArrayList<WeakReference<double[]>> topValues = null;
		private Scope() {
		}
		public void enter() {
			values.push(topValues = new ArrayList<WeakReference<double[]>>());
		}
		public void exit() {
			for(WeakReference<double[]> ref : topValues) {
				double[] value = ref.get();
				if(value!=null)
					release(value);
			}
			topValues.clear();
			topValues = values.pop();
		}
		public void register(double[] value) {
			if(topValues!=null)
				topValues.add(new WeakReference<double[]>(value));
		}
		public void unregister(double[] value) {
			if(topValues!=null) {
				for(WeakReference<double[]> ref : topValues)
					if(ref.get()==value) {
						topValues.remove(ref);
						break;
					}
			}
		}
	}
	
	private static HashMap<Integer, Scope> scopes = new HashMap<Integer, Scope>();
	
	public static Scope scope() {
		int threadId = ThreadPool.getCurrentThreadId();
		Scope ret;
		synchronized(scopes) {
			ret = scopes.get(threadId);
		}
		if(ret==null) {
			synchronized(scopes) {
				scopes.put(threadId, ret = new Scope());
			}
		}
		return ret;
	}
	
	private static class BoundAllocation {
		private SoftReference<double[]> memory;
		public WeakReference<Object> boundObject;
		
		public BoundAllocation(int length, Object boundObject) {
			memory = new SoftReference<double[]>(new double[length]);
			this.boundObject = new WeakReference<Object>(boundObject);
		}
		public void changeBoundObject(Object boundObject) {
			this.boundObject = new WeakReference<Object>(boundObject);
		}
		public boolean isReusable() {
			return boundObject.get()==null && memory.get()!=null;
		}
		public boolean isInvalid() {
			return memory.get() == null;
		}
		public double[] value() {
			return memory.get();
		}
	}
	
	private static HashMap<Integer, ArrayList<BoundAllocation>> allocated = new HashMap<Integer, ArrayList<BoundAllocation>>();
	private static WeakHashMap<double[], BoundAllocation> bounded = new WeakHashMap<double[], BoundAllocation>();
	
	public synchronized static double[] allocate(int length, Object boundTo) {
		ArrayList<BoundAllocation> search = allocated.get(length);
		if(search==null)
			allocated.put(length, search = new ArrayList<BoundAllocation>());
		ArrayList<BoundAllocation> toDelete = null;
		for(BoundAllocation ref : search) {
			if(ref.isReusable()) {
				double[] ret = ref.value();
				for(int i=0;i<ret.length;i++)
					ret[i] = 0;
				ref.changeBoundObject(boundTo);
				return ret;
			}
			if(ref.isInvalid()) {
				if(toDelete==null)
					toDelete = new ArrayList<BoundAllocation>();
				toDelete.add(ref);
			}
		}
		if(toDelete!=null)
			search.removeAll(toDelete);
		BoundAllocation ret = new BoundAllocation(length, boundTo);
		search.add(ret);
		bounded.put(ret.value(), ret);
		scope().register(ret.value());
		return ret.value();
	}
	
	public static synchronized void release(double[] value) {
		BoundAllocation ref = bounded.get(value);
		if(ref!=null)
			ref.changeBoundObject(null);
	}
}
