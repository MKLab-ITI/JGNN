package mklab.JGNN.core.util;

import java.util.ArrayList;

public class Sort { 
	public static int[] sortedIndexes(double A[]) {
		int[] indexes = new int[A.length];
		for(int i=0;i<A.length;i++)
			indexes[i] = i;
		merge_sort(A, indexes, 0, A.length-1);
		return indexes;
	}
	
	public static int[] sortedIndexes(ArrayList<Double> A) {
		double[] Alist = new double[A.size()];
		for(int i=0;i<A.size();i++)
			Alist[i] = A.get(i);
		return sortedIndexes(Alist);
	}
	
    private static int partition(double A[], int indexes[], int low, int high) { 
        double pi = A[high];  
        int i = (low-1); // smaller element index   
        for (int j=low; j<high; j++) { 
            if (A[j] <= pi) { 
                i++; 
                double tempA = A[i]; 
                A[i] = A[j]; 
                A[j] = tempA; 
                int tempI = indexes[i];
                indexes[i] = indexes[j]; 
                indexes[j] = tempI; 
            } 
        }
        double tempA = A[i+1]; 
        A[i+1] = A[high]; 
        A[high] = tempA; 
        int tempI = indexes[i+1]; 
        indexes[i+1] = indexes[high]; 
        indexes[high] = tempI; 
        return i+1; 
    } 
 
    @SuppressWarnings("unused")
	private static void quick_sort(double A[], int indexes[], int low, int high) { 
        if (low < high) { 
        	while(high>low && A[indexes[high]]>A[indexes[high-1]])
        		high -= 1;
            int pi = partition(A, indexes, low, high); 
            quick_sort(A, indexes, low, pi-1); 
            quick_sort(A, indexes, pi+1, high); 
        } 
    }
    
    
    private static void merge(double A[], int indexes[], int l, int m, int r) 
    { 
        // Find sizes of two subarrays to be merged 
        int n1 = m - l + 1; 
        int n2 = r - m; 
  
        /* Create temp arrays */
        int L[] = new int[n1]; 
        int R[] = new int[n2]; 
  
        /*Copy data to temp arrays*/
        for (int i = 0; i < n1; ++i) 
            L[i] = indexes[l + i]; 
        for (int j = 0; j < n2; ++j) 
            R[j] = indexes[m + 1 + j]; 
  
        /* Merge the temp arrays */
  
        // Initial indexes of first and second subarrays 
        int i = 0, j = 0; 
  
        // Initial index of merged subarry array 
        int k = l; 
        while (i < n1 && j < n2) { 
            if (A[L[i]] <= A[R[j]]) { 
            	indexes[k] = L[i]; 
                i++; 
            } 
            else { 
            	indexes[k] = R[j]; 
                j++; 
            } 
            k++; 
        } 
  
        /* Copy remaining elements of L[] if any */
        while (i < n1) { 
        	indexes[k] = L[i]; 
            i++; 
            k++; 
        } 
  
        /* Copy remaining elements of R[] if any */
        while (j < n2) { 
        	indexes[k] = R[j]; 
            j++; 
            k++; 
        } 
    } 
 
    static void merge_sort(double A[], int indexes[], int l, int r) { 
        if (l < r) { 
            // Find the middle point 
            int m = (l + r) / 2;   
            // Sort first and second halves 
            merge_sort(A, indexes, l, m); 
            merge_sort(A, indexes, m + 1, r); 
            // Merge the sorted halves 
            merge(A, indexes, l, m, r); 
        } 
    } 
}
