package graphClassification;

import java.util.Arrays;

import mklab.JGNN.adhoc.ModelBuilder;
import mklab.JGNN.adhoc.ModelTraining;
import mklab.JGNN.adhoc.parsers.LayeredBuilder;
import mklab.JGNN.adhoc.train.AGFTraining;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.ThreadPool;
import mklab.JGNN.nn.Loss;
import mklab.JGNN.nn.Model;
import mklab.JGNN.nn.initializers.XavierNormal;
import mklab.JGNN.nn.loss.Accuracy;
import mklab.JGNN.nn.loss.CategoricalCrossEntropy;
import mklab.JGNN.nn.loss.report.VerboseLoss;
import mklab.JGNN.nn.optimizers.Adam;
import mklab.JGNN.nn.optimizers.BatchOptimizer;

/**
 * 
 * @author github.com/gavalian
 * @author Emmanouil Krasanakis
 */
public class SortPooling {
    
    public static void main(String[] args){
        long reduced = 5;  // input graphs need to have at least that many nodes, lower values decrease accuracy
        long hidden = 8;  // since this library does not use GPU parallelization, many latent dims reduce speed

        ModelBuilder builder = new LayeredBuilder()        
                .var("A")  
                .config("features", 1)
                .config("classes", 2)
                .config("reduced", reduced)
                .config("hidden", hidden)
                .config("reg", 0.005)
                .layer("h{l+1}=relu(A@(h{l}@matrix(features, hidden, reg))+vector(hidden))") 
                .layer("h{l+1}=relu(A@(h{l}@matrix(hidden, hidden, reg))+vector(hidden))") 
                .concat(2) // concatenates the outputs of the last 2 layers
                .config("hiddenReduced", hidden*2*reduced)  // 2* due to concatenation
                .operation("z{l}=sort(h{l}, reduced)")  // currently, the parser fails to understand full expressions within next step's gather, so we need to create this intermediate variable
                .layer("h{l+1}=reshape(h{l}[z{l}], 1, hiddenReduced)") //
                .layer("h{l+1}=h{l}@matrix(hiddenReduced, classes)")
                .layer("h{l+1}=softmax(h{l}, dim: 'row')")
                //.layer("h{l+1}=softmax(sum(h{l}@matrix(hiddenReduced, classes), row))")//this is mean pooling to replace the above sort pooling
                .out("h{l}");       
        TrajectoryData dtrain = new TrajectoryData(8000);
        TrajectoryData dtest = new TrajectoryData(2000);
        
        ModelTraining trainer = new AGFTraining()
        		.setGraphs(dtrain.graphs)
        		.setNodeFeatures(dtrain.features)
        		.setGraphLabels(dtrain.labels)
        		.setValidationSplit(0.2)
        		.setEpochs(300)
        		.setOptimizer(new Adam(0.001))
        		.setLoss(new CategoricalCrossEntropy())
        		.setNumBatches(10)
        		.setParallelizedStochasticGradientDescent(true)
        		.setValidationLoss(new VerboseLoss(new CategoricalCrossEntropy(), new Accuracy()));
        
        Model model = builder.getModel()
        		.init(new XavierNormal())
        		.train(trainer);
            
        double acc = 0.0;
        for(int graphId=0; graphId<dtest.graphs.size(); graphId++) {
            Matrix adjacency = dtest.graphs.get(graphId);
            Matrix features= dtest.features.get(graphId);
            Tensor graphLabel = dtest.labels.get(graphId);
            if(model.predict(Arrays.asList(features, adjacency)).get(0).argmax()==graphLabel.argmax())
                acc += 1;
        }
        System.out.println("Test accuracy " + acc/dtest.graphs.size());
    }
}
