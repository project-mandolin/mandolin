# XGBoost

Mandolin provides a wrapper for XGBoost.  The standard Mandolin file format is used (rather than XGBoost's native
format). The wrapper also facilitates use of model selection over XGBoost hyper-parameters. (see . . . )

## Splice Example

A small excerpt of the full Splice dataset is used here to demonstrate the utiltiy of XGBoost
for modeling problems better captured with a non-linear classifier. The example can be run via:

    cd $MANDOLIN/examples/splice
    java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g \
        -cp ../../dist/mandolin-mx-0.3.6.jar --conf splice.conf

The default `splice.conf` file in this example runs Mandolin in `train-test` mode. Here, with a small dataset,
the training file is used as the test file.  This is a big *no-no*, generally, but can be helpful to see
how well the model is fitting the training data. We can perform a more meaningful experiment by running
cross validation on the training data. This is triggered by simply setting the `test-file` to null which can
be done by editing the `splice.conf` file or overriding on the command-line as follows:

     java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g \
        -cp ../../dist/mandolin-mx-0.3.6.jar --conf splice.conf mandolin.xg.test-file=null

The progress and evaluation metrics across 5 folds are logged to the console after each epoch/round/iteration
of training.

### Training a model

Running in `train-test` mode is good for experimentation, whether using cross validation or a validation/test
dataset. We can train a model to be saved for making subsequent predictions by running in
`train` mode.

    java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g \
        -cp ../../dist/mandolin-mx-0.3.6.jar --conf splice.conf mandolin.mode=train

This will save a model in the file specified via the configuration attribute `mandolin.xg.model-file`.
If certain configruation options are set to non-null values, some side-effects of the training
process may be realized. In the example here, the attribute `mandolin.xg.feature-importance-file`
is set to non-null and Mandolin will provide each feature with it's associated Gain
according to XGBoost in file provided.

### Making Predictions and Evaluating

The trained model can be used to make predictions on a held out test dataset and provide
an evaluation by running Mandolin in `predict-eval` mode.  In this case, we could evaluate
the model on the training data (again, a held out test set should be used in practice) via:

    java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g \
        -cp ../../dist/mandolin-mx-0.3.6.jar --conf splice.conf mandolin.mode=predict-eval \
	mandolin.xg.prediction-file=outputs.1

This will take the trained model file (specified via `mandolin.xg.model-file` in `splice.conf`)
and apply it to the labeled instances in `mandolin.xg.test-file`. The resulting predictions
are placed in the file `outputs.1` as specified via
`mandolin.xg.prediction-file`. As this is a binary classification task, the prediction file
format has three columns:

    ID, response, value

The `ID` refers to an optional unique ID for each test instance that can be specified by including
a `#` symbol followed by the ID on the input file.   This defaults to -1 if no IDs are specified,
as is the case here.  The `response` field provides the probability of the instance being
positive according to the model. Note that the positive label (in this example `+1`) should
be the first label in the `mandolin.xg.labels` file. The `value` field provides the "ground truth" label
for the instance as it appears in the input file.

### Making Predictions on Novel Data without Ground Truth

Tuning ML algorithms is done on training and test datasets with available ground truth to ascertain
accuracy/performance. Ultimately models are designed to be used to make predictions on data
without any available ground truth.  Mandolin models can be used to make predictions by running
the software in `predict` mode. This assumes that the test file in `mandolin.xg.test-file` includes
feature vectors (just like the training and test instances with labels), but that the left-most
column is removed - i.e. no label is provided. An example file without labels is provided with the example
called `rawtest.vecs`. Our trained model can be used to make predictions on these intances by invoking:

    java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g \
        -cp ../../dist/mandolin-mx-0.3.6.jar --conf splice.conf mandolin.mode=predict \
	mandolin.xg.test-file=rawtest.vecs mandolin.xg.prediction-file=outputs.2

Since the ground truth labels are not provided here, the resulting prediction file, `outputs.2`
will not contain the final `value` column.

While using `predict` mode can be convenient for batch processing data, using the learnt model
to make predictions programmatically is often better for embedding within an application.
See (...) for information on using Mandolin-trained models programmatically via an API.