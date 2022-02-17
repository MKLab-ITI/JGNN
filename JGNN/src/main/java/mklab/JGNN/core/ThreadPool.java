package mklab.JGNN.core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * This class provides thread execution pool utilities while keeping track of thread
 * identifiers for use by thread-specific {@link NNOperation}.
 * 
 * @author Emmanouil Krasanakis
 */
public class ThreadPool {
	private HashMap<Thread, Integer> threadIds = new HashMap<Thread, Integer>();
	private HashSet<Integer> usedIds = new HashSet<Integer>();
	private ThreadPoolExecutor executor;
	private int maxThreads;

	private static ThreadPool instance = new ThreadPool(Runtime.getRuntime().availableProcessors());
	public static ThreadPool getInstance() {
		return instance; 
	}
	
	protected ThreadPool(int maxThreads) {
		this.maxThreads = maxThreads;
		executor = null;
	}
	protected int getUnusedId() {
		for(int i=0;i<maxThreads;i++)
			if(!usedIds.contains(i))
				return i;
		return -1;//new RuntimeException("Could not retrieve an unused thread id");
	}
	public synchronized void submit(Runnable runnable) {
		Thread thread = new Thread() {
			@Override
			public void run() {
				synchronized(threadIds) {
					int threadId = getUnusedId();
					if(threadId==-1) 
						throw new RuntimeException("Tried to instantiate thread without an available id");
					threadIds.put(Thread.currentThread(), threadId);
					usedIds.add(threadId);
				}
				runnable.run();
				synchronized(threadIds) {
					int threadId = getCurrentThreadId();
					threadIds.remove(this);
					usedIds.remove(threadId);
				}
			}
		};
		if(executor==null)
			executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(maxThreads);
		executor.submit(thread);
	}
	/**
	 * Retrieves a unique integer indicating the currently running thread.
	 * @return An integer id.
	 */
	public static Integer getCurrentThreadId() {
		Integer ret = getInstance().threadIds.get(Thread.currentThread());
		return ret==null?-1:(int)ret;
	}
	/**
	 * Waits until all threads in the pool have finished.
	 */
	public void waitForConclusion() {
		executor.shutdown();
		try {
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES);
		}
		catch (InterruptedException e) {
			e.printStackTrace();
		}
		finally {
			executor = null;
		}
	}
}
