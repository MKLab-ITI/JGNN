import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.VectorizedMatrix;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.tensor.SparseTensor;
import mklab.JGNN.core.tensor.VectorizedTensor;

public class Benchmarks {
	private static void compareTensors() {
		int n = 100000;
		System.out.println("## Adding tensors of "+n+" elements");
		{
			Tensor a = new SparseTensor(n).setToRandom(); 
			Tensor b = new SparseTensor(n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<30;++i) {
				Tensor c = a.add(b);
			}
			long toc = System.currentTimeMillis();
			System.out.println("Sparse: "+(toc-tic)/1000.0/30+" sec");
		}
		{
			Tensor a = new DenseTensor(n).setToRandom(); 
			Tensor b = new DenseTensor(n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<30;++i) {
				Tensor c = a.add(b);
			}
			long toc = System.currentTimeMillis();
			System.out.println("Dense: "+(toc-tic)/1000.0/30+" sec");
		}
		if(Tensor.vectorization)
		{
			Tensor a = new VectorizedTensor(n).setToRandom();
			Tensor b = new VectorizedTensor(n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<30;++i) {
				Tensor c = a.add(b);
			}
			long toc = System.currentTimeMillis();
			System.out.println("Vectorized: "+(toc-tic)/1000.0/30+" sec");
		}
		else
			System.out.println("Vectorized: not supported");
	}
	
	private static void compareMatrices() {
		int m = 16;
		int n = 16;
		System.out.println("## Multiplying matrices where the first is "+m+"x"+n+"");
		{
			Tensor a = new SparseMatrix(m, n).setToRandom(); 
			Tensor b = new SparseMatrix(n, n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<5;++i) {
				Tensor c = a.cast(Matrix.class).matmul(b.cast(Matrix.class));
			}
			long toc = System.currentTimeMillis();
			System.out.println("Sparse: "+(toc-tic)/1000.0/5+" sec");
		}
		
		{
			Tensor a = new DenseMatrix(m, n).setToRandom(); 
			Tensor b = new DenseMatrix(n, n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<5;++i) {
				Tensor c = a.cast(Matrix.class).matmul(b.cast(Matrix.class));
			}
			long toc = System.currentTimeMillis();
			System.out.println("Dense: "+(toc-tic)/1000.0/5+" sec");
		}

		{
			Tensor a = new VectorizedMatrix(m, n).setToRandom(); 
			Tensor b = new VectorizedMatrix(n, n).setToRandom();
			long tic = System.currentTimeMillis();
			for(int i=0;i<5;++i) {
				Tensor c = a.cast(Matrix.class).matmul(b.cast(Matrix.class));
			}
			long toc = System.currentTimeMillis();
			System.out.println("Vectorized: "+(toc-tic)/1000.0/5+" sec");
		}
	}

	public static void main(String[] args) {
		compareTensors();
		//compareMatrices();
	}

}
