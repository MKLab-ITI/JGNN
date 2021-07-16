package mklab.JGNN.examples;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.matrix.DenseMatrix;

public class Primitives {
	public static void main(String[] args) throws Exception {
		Matrix matrix = new DenseMatrix(2,2).put(0,0,1).put(0,1,2);
		System.out.println(matrix.toString());
		System.out.println(matrix.asTransposed().selfAdd(1).toString());
		System.out.println(matrix.toString());
	}

}
