package mklab.JGNN.core;

import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * This class provides thread execution pool utilities while keeping track of
 * thread identifiers for use by thread-specific
 * {@link mklab.JGNN.nn.NNOperation}. Threads scheduling relies on Java's
 * {@link ThreadPoolExecutor}.
 * 
 * @author Emmanouil Krasanakis
 */
public class ThreadPool {
	private static HashMap<Thread, Integer> threadIds = new HashMap<Thread, Integer>();
	private static HashSet<Integer> usedIds = new HashSet<Integer>();
	private ThreadPoolExecutor executor;
	private int maxThreads;

	private static ThreadPool instance = new ThreadPool(Runtime.getRuntime().availableProcessors());

	/**
	 * Retrieves the singleton {@link ThreadPool} instance used by JGNN.
	 * 
	 * @return A {@link ThreadPool}.
	 */
	public static ThreadPool getInstance() {
		return instance;
	}

	protected ThreadPool(int maxThreads) {
		this.maxThreads = maxThreads;
		executor = null;
	}

	protected int getUnusedId() {
		for (int i = 0; i < maxThreads; i++)
			if (!usedIds.contains(i))
				return i;
		return -1;// new RuntimeException("Could not retrieve an unused thread id");*/
	}

	/**
	 * Submits a runnable to be executed at some future point by a thread, for
	 * example via
	 * <code>ThreadPool.getInstance().submit(new Runnable(){public void run(){...}});</code>.
	 * 
	 * @param runnable A Java {@link Runnable}.
	 * @see #waitForConclusion()
	 */
	public synchronized void submit(Runnable runnable) {
		Thread thread = new Thread() {
			@Override
			public void run() {
				Thread current = Thread.currentThread();
				int threadId;
				synchronized (threadIds) {
					threadId = getUnusedId();
					if (threadId == -1)
						throw new RuntimeException("Tried to instantiate thread without an available id");
					//System.out.println("Starting thread #"+threadId);
					threadIds.put(current, threadId);
					usedIds.add(threadId);
				}
				runnable.run();
				synchronized (threadIds) {
					threadIds.remove(current);
					usedIds.remove(threadId);
					//System.out.println("Ending thread #"+threadId);
				}
			}
		};
		if (executor == null)
			executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(maxThreads);
		executor.submit(thread);
	}

	/**
	 * Retrieves a unique integer indicating the currently running thread.
	 * 
	 * @return An integer id.
	 */
	public static Integer getCurrentThreadId() {
		Integer ret;
		Thread current = Thread.currentThread();
		synchronized (threadIds) {
			ret = threadIds.get(current);
		}
		return ret == null ? -1 : (int) ret;
	}

	/**
	 * Waits until all threads in the pool have finished. This concludes only if all
	 * submitted runnable conclude.
	 * 
	 * @see #submit(Runnable)
	 */
	public void waitForConclusion() {
		executor.shutdown();
		try {
			executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MINUTES);
		} catch (InterruptedException e) {
			e.printStackTrace();
		} finally {
			executor = null;
		}
	}
}
