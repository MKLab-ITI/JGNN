package mklab.JGNN;

public class LSTMTest {
	/*@Test
	public void LSTMShouldLearnTimeSeries() {
		Utils.development = true;
		Optimizer updateRule = new Optimizer.Regularization(new Optimizer.Adam(1), 1.E-12);
		LSTM lstm = new LSTM(updateRule, 1, 1);
		int depth = 2;
		
		Tensor[] series = {Tensor.fromDouble(.11),
				Tensor.fromDouble(.12),
				Tensor.fromDouble(.13),
				Tensor.fromDouble(.14),
				Tensor.fromDouble(.15),
				Tensor.fromDouble(.16),
				Tensor.fromDouble(.17),
				Tensor.fromDouble(.18),
				Tensor.fromDouble(.19),
				Tensor.fromDouble(.20),
				Tensor.fromDouble(.21),
				Tensor.fromDouble(.22),
				Tensor.fromDouble(.23),
				Tensor.fromDouble(.24),
				Tensor.fromDouble(.25),
				Tensor.fromDouble(.26),
				Tensor.fromDouble(.27),
				Tensor.fromDouble(.28),
				Tensor.fromDouble(.29),
				Tensor.fromDouble(.30),
		};
		double test_error = 0;
		double total_error = 0;
		for(int epoch=0;epoch<1000;epoch++) {
			total_error = 0;
			test_error = 0;
			lstm.startTape();
			for(int i=0;i<series.length-depth-1;i++) {
				Tensor[] inputs = new Tensor.DenseTensor[depth];
				for(int j=0;j<depth;j++)
					inputs[j] = series[i+j];
				total_error += lstm.train(inputs, series[i+depth]) / (series.length-depth-1);
				//System.out.println(lstm.predict(inputs).toDouble()+" vs "+series[i+depth].toDouble());
			}
			lstm.endTape();
			{
				int i = series.length-depth-1;
				Tensor[] inputs = new Tensor.DenseTensor[depth];
				for(int j=0;j<depth;j++)
					inputs[j] = series[i+j];
				test_error += Math.abs(series[i+depth].toDouble()-lstm.predict(inputs).toDouble());
				//System.out.println(lstm.predict(inputs).toDouble()+" vs "+series[i+depth].toDouble());
			}
			//System.out.println("LSTM Error "+total_error+" test_error "+test_error);
		}
		System.out.println("LSTM Training Error "+total_error+" and Test Error "+test_error);
		Assert.assertEquals(0., test_error, 0.015);
	}
	
	
	@Test
	public void LSTMShouldLearnVectorSeries() {
		Utils.development = true;
		Optimizer updateRule = new Optimizer.Regularization(new Optimizer.Adam(1), 1.E-12);
		LSTM lstm = new LSTM(updateRule, 2, 1);
		int depth = 2;
		
		Tensor[] series = {
				new Tensor.DenseTensor(".1, .1"),
				new Tensor.DenseTensor(".1, .2"),
				new Tensor.DenseTensor(".2, .3"),
				new Tensor.DenseTensor(".3, .5"),
				new Tensor.DenseTensor(".5, .8"),
				new Tensor.DenseTensor(".8, 1.3"),
				new Tensor.DenseTensor(".1, .1"),
				new Tensor.DenseTensor(".1, .2"),
				new Tensor.DenseTensor(".2, .3"),
				new Tensor.DenseTensor(".3, .5"),
				new Tensor.DenseTensor(".5, .8"),
				new Tensor.DenseTensor(".8, 1.3")
		};
		Tensor[] output = {
				new Tensor.DenseTensor(".2"),
				new Tensor.DenseTensor(".3"),
				new Tensor.DenseTensor(".5"),
				new Tensor.DenseTensor(".8"),
				new Tensor.DenseTensor("1.3"),
				new Tensor.DenseTensor(".2"),
				new Tensor.DenseTensor(".3"),
				new Tensor.DenseTensor(".5"),
				new Tensor.DenseTensor(".8"),
				new Tensor.DenseTensor("1.3")
		};
		double test_error = 0;
		double total_error = 0;
		for(int epoch=0;epoch<1000;epoch++) {
			total_error = 0;
			test_error = 0;
			lstm.startTape();
			for(int i=0;i<series.length-depth-1;i++) {
				Tensor[] inputs = new Tensor.DenseTensor[depth];
				for(int j=0;j<depth;j++)
					inputs[j] = series[i+j];
				total_error += lstm.train(inputs, output[i+depth-1]) / (series.length-depth-1);
				System.out.println(lstm.predict(inputs).toString()+" vs "+output[i+depth-1].toString());
			}
			lstm.endTape();
			{
				int i = series.length-depth-2;
				Tensor[] inputs = new Tensor.DenseTensor[depth];
				for(int j=0;j<depth;j++)
					inputs[j] = series[i+j];
				test_error += output[i+depth-1].subtract(lstm.predict(inputs)).norm();
				System.out.println(lstm.predict(inputs).toString()+" vs "+output[i+depth-1].toString());
			}
			//System.out.println("LSTM Error "+total_error+" test_error "+test_error);
		}
		System.out.println("LSTM Training Error "+total_error+" and Test Error "+test_error);
		Assert.assertEquals(0., test_error, 0.015);
	}*/
}
