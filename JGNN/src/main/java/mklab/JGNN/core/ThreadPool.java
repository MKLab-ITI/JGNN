package mklab.JGNN.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 * This class provides a thread manager that automatically schedules threaded runnables
 * so that the total number of threads running at a given time does not exceed the number
 * of available processors.
 * 
 * @author Emmanouil Krasanakis
 */
public class ThreadPool {
	private HashMap<Thread, Integer> threadIds = new HashMap<Thread, Integer>();
	private HashSet<Integer> usedIds = new HashSet<Integer>();
	private int maxThreads;
	private static ThreadPool instance = new ThreadPool(Runtime.getRuntime().availableProcessors()-1);
	
	/**
	 * Retrieves a singleton thread pool
	 * with a number of threads equal to the number of available processors minus 1.
	 * @return A ThreadPool instance.
	 */
	public static ThreadPool getInstance() {
		return instance; 
	}
	
	protected ThreadPool(int maxThreads) {
		this.maxThreads = maxThreads;
	}
	
	/**
	 * Waits until at least one thread is available.
	 */
	public void waitForVacancy() {
		while(threadIds.size()>=maxThreads) {
			try {
				Thread.sleep(1);
			}
			catch (Exception e){
			}
		}
	}
	protected int getUnusedId() {
		for(int i=0;i<maxThreads;i++)
			if(!usedIds.contains(i))
				return i;
		return -1;//new RuntimeException("Could not retrieve an unused thread id");
	}
	/**
	 * Waits until space for a new thread is made and then executes the provided runnable.
	 * @param runnable The runnable to execute.
	 * @see #waitForVacancy()
	 */
	public synchronized void start(Runnable runnable) {
		Thread thread = new Thread() {
			@Override
			public void run() {
				runnable.run();
				int threadId = threadIds.get(this);
				threadIds.remove(this);
				usedIds.remove(threadId);
			}
		};
		int threadId = getUnusedId();
		while(threadId==-1) {
			waitForVacancy();
			threadId = getUnusedId();
		}
		threadIds.put(thread, threadId);
		usedIds.add(threadId);
		thread.start();
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
		for(Thread thread : new ArrayList<Thread>(threadIds.keySet())) {
			try {
				thread.wait();
			}
			catch(Exception e) {
			}
		}
	}
	/**
	 * Retrieves the maximum number of threads allowed to run at the same time.
	 * @return The maximum number of threads.
	 */
	public int getMaxThreads() {
		return maxThreads;
	}
}
