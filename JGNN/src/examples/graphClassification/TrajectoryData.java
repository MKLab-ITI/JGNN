package graphClassification;


import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.tensor.DenseTensor;

/**
 * This class generates trajectory graph labels. 
 * Graph nodes correspond to points in time.
 * 
 * @author github.com/gavalian (author)
 * @author Emmanouil Krasanakis (editing)
 */
public class TrajectoryData {
    public List<Matrix> graphs = new ArrayList<>();
    public List<Matrix> features = new ArrayList<>();
    public List<Tensor> labels = new ArrayList<>();
    
    private Random r = new Random();
    protected double[] createTrajectory(int size){
        double v = 40.0;
        double angle = r.nextDouble()*40+15;
        double g = 9.8;
        double vx = v*Math.cos(Math.toRadians(angle));
        double vy = v*Math.sin(Math.toRadians(angle));
        double[] f = new double[size];
        double x = 4;
        for(int counter=0; counter<size; counter++){
        	x+= 4;
            double t = x/vx;
            double y = vy*t - g*t*t;
            f[counter] = y/28.0;
        }
        return f;
    }
    
    public TrajectoryData(int numGraphs){
    	createGraphs(numGraphs);
    }
    
    protected Matrix createTimeAdjacency(int size){
        return new SparseMatrix(size,size)
        			//.setDiagonal(0, 1.) // self edge
        			//.setDiagonal(-1, 1.) // edge from subsequent to previous time
        			.setDiagonal(1, 1.);
    }
    
    protected void createGraphs(int numGraphs){
        for(int i = 0; i < numGraphs; i++){
        	int size = 10+(int)(2*Math.random());
            Matrix mt = createTimeAdjacency(size);
            Matrix mf = createTimeAdjacency(size);
            
            double[] trajectory = createTrajectory(size);
            
            Matrix ff = new DenseTensor(trajectory).asColumn().toDense();
            Matrix ft = new DenseTensor(trajectory).asColumn().toDense();
            //int  which = r.nextInt(8)+1;
            //ff.put(which, 0, traj[which]/2.0);
            for(int j = 0; j < 3; j++){
                int w2 = r.nextInt(8)+1;
                ff.put(w2, 0, 0.0);
            }
            this.graphs.add(mt);
            this.graphs.add(mf);
            
            this.features.add(ft);
            this.features.add(ff);
            
            this.labels.add(new DenseTensor(2).put(0, 1.0).asRow());
            this.labels.add(new DenseTensor(2).put(1, 1.0).asRow());
        }
    }
}
