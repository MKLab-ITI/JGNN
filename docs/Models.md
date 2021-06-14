# Models and builders

## Table of Contents

1. [JGNN Expressions](#jgnn-expressions)
2. [JGNN Models](#jgnn-models)
3. [Symbolic Model Definition](#symbolic-model-definition)
4. [Learning Parameters](#learning-parameters)
5. [Neural Network Examples](#neural-network-examples)
6. [Multithread Batch Learning](#multithread-batch-learning)

## JGNN Expressions

The base structure used to define machine learning operations is the `mklab.JGGN.core.NNOperation` abstract class.
This is implemented by common mathematical operations, which are presented in the following table. Operation instances
can be added to inputs of other operations through the `addInput(NNOperation)` method of the latter. Starting points
of operations are variables, constants and parameters, whose differences will be discussed later.

:bulb: The hustle of learning to write expressions is removed with [Symbolic model definition](#symbolic-model-definition).
You can safely skip to that segment to learn how to write machine learning models without the tedious definitions of intermediate steps explained here.

|Operator| Constructor | Number of inputs  |
| --- | --- | --- |
| + | mklab.JGNN.core.operations.Add()      | 2 |
| * | mklab.JGNN.core.operations.Multiply() | 2 |
| @ | mklab.JGNN.core.operations.MatMul()   | 2 |
| 1-x | mklab.JGNN.core.operations.Complement()   | 1 |
| log | mklab.JGNN.core.operations.Log() | 1 |
| variable | mklab.JGNN.core.operations.Variable() | 0 |
| constant | mklab.JGNN.core.operations.Constant(tensor) | 0 |
| parameter | mklab.JGNN.core.operations.Parameter(tensor) | 0 |
| relu | mklab.JGNN.core.activations.Relu() | 1 |
| relu | mklab.JGNN.core.activations.Tanh() | 1 |
| relu | mklab.JGNN.core.activations.Sigmoid() | 1 |
| lrelu | mklab.JGNN.core.activations.LRelu() | 2 |
| lrelu | mklab.JGNN.core.activations.PRelu() | 2 |

:Warn: In principle, the `addInput` should be called a number of times equal to the number of operator arguments for each operator.
It is defined for the sake of convenience, for example to initialize operators at different parts of the code than the one linking them.

:Warn: Detailed error checking of JGNN operations is under development.

For example, the expression *y=log(2x+1)* can be constructed with the following code:

```java
	Variable x = new Variable();
	Constant c1 = new Constant(Tensor.fromDouble(1)); // holds the constant "1"
	Constant c2 = new Constant(Tensor.fromDouble(2)); // holds the constant "2"
	NNOperation mult = new Multiply().addInput(x).addInput(c2);
	NNOperation add = new Add().addInput(mult).addInput(c1);
	NNOperation y = new Log().addInput(add);
}
```

## JGNN Models

Constructed expressions can be organized into machine learning models. Models are implemented by the class `mklab.JGNN.core.Model`
and defining them is as simple as marking the input variables with the method `Model addInput(Variable)` and output operations
with the method `Model addOutput(NNOperation)`. For example, constructing a model holding the previous expression is as simple as writing
`Model model = new Model().addInput(x).addOutput(y)`. Potential backpropagation machine learning operations are automatically handled
by models.

Running the model once to create outputs can be achieved with `Tensor Model.predict(Tensor...)` method. This takes as input one or more
comma-separated tensors to pass into the model. 
If the number of inputs is dynamically created, an overloaded version of the same method supports an array list of input tensors
`Tensor Model.predict(ArrayList<Tensor>)`. 

:warn: Input tensor order should be the same to the order variables were added to the model.

Obtaining the last value of intermediate (i.e. non-ouput) operations *after* the run can achieved with the `Tensor NNOperation.getPrediction()` method. To sum up with an example, running a model of the previously defined *y=log(2x+1)* for *x=2* and printing both the value of *y* (approx. 1.61) and the value inside the logarithm (5) can be achieved with with the following code:

```java	
	Model model = new Model().addInput(x).addOutput(log);
	System.out.println(model.predict(Tensor.fromDouble(2)));
	System.out.println(add.getPrediction());
```


## Symbolic model definition


```java
ModelBuilder modelBuilder = new ModelBuilder()
		.var("x") // first argument
		.constant("a", Tensor.fromDouble(2))
		.constant("b", Tensor.fromDouble(1))
		.operation("yhat = a*x+b")
		.out("yhat")
		.print() // comment out this line to not print the model
		;
System.out.println(modelBuilder.getModel().predict(Tensor.fromDouble(2)));
```

## Learning parameters

Examples up to this point were limited to using constant and variable data. However, machine learning
tasks typically introduce the notion of *parameters* as constants whose values can be learned to optimize
learning objectives, such as making model output values as close as possible to desired ones.

Parameter operations can be instantiated with the constructor `new mklab.JGGN.core.Parameter(Tensor initialValue)`,
where their initial values or provided. Model builder parameters need to be defined before they are used
by operations and can be symbolically defined with the method
`ModelBuilder ModelBuilder.param(String name, Tensor initialValue)`.

Approximating ideal parameter values for a model requires three steps: a) selecting an optimization scheme responsible for
updating parameters based on backpropagated errors, b) selecting a loss function quantifying how much model outputs deviate
from optimal ones and c) repeatedly calling one of the model's overloaded `Model.trainSample` methods for a number of epochs. 
For the sake of simplicity, consider a single sample before we discuss how to handle multiple ones.


```java
ModelBuilder modelBuilder = new ModelBuilder()
		.var("x") // first argument
		.var("y") // second argument
		.param("a", Tensor.fromDouble(1))
		.param("b", Tensor.fromDouble(0))
		.operation("yhat = a*x+b")
		.operation("error = (y-yhat)*(y-yhat)")
		.out("error")
		.print();
Optimizer optimizer = new Adam(0.1);
// when no output is passed to training, the output is considered to be an error
for(int i=0;i<200;i++)
	modelBuilder.getModel().trainSample(optimizer, Arrays.asList(new DenseTensor(1,2,3,4,5), new DenseTensor(3,5,7,9,11)));
//run the wrapped model and obtain an internal variable prediction
System.out.println(modelBuilder.runModel(Tensor.fromDouble(2), Tensor.fromDouble(0)).get("yhat").getPrediction());
```

:bulb: Examples with multiple features should either be organized into sparce matrices or be fed one-by-one to learners
through a [batch-learning](#multithread-batch-learning) scheme. Specifically for graph neural networks, computation
speed benefits tremendously from organizing all node features into one matrix and simultaneously passing this through
graph convolutional layers.

## Neural Network Examples


## Multithread Batch Learning