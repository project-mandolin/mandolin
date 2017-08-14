# File Formats

Mandolin includes a standard file format for representing training and test instances that
is based on the sparse vector representation used by libSVM. 

For classification problems, the input  has the following
form for each training/test instances:

    <label> <feature_1>:<value> <feature_2>:<value> ... <feature_N>:<value> 

Each line represents a separate training instance.
The `<label>` value is a string that denotes the label/class of the training instance.
Each `<feature_i>` is either a name of a unique feature in the model or an index that refers
to which component in the input vector the feature corresponds to.  This input format lends
itself well to *sparse* representations where the total number of features in the model
is large, but for any given instance a relatively small number of features will have a non-zero
value.

## Dense Vector Representation

This representation uses a [dense vector](../api/index.html#org.mitre.mandolin.util.DenseTensor1) 
as the model inputs.  It reads the input assuming that each `<feature_i>` denoting a feature
corresponds to index of the feature in the dense vector. 
This requires that the dimension of the input space be specified a priori.
This is done by setting the configuration `mandolin.app.trainer.dense-vector-size` to a positive 
integer value. For example, if `mandolin.app.trainer.dense-vector-size=4` an input file
with feature representations has dense vectors corresponding to those to the right.

     1:0.2 3:0.4         # ==> [0.2 0.0 0.4 0.0]
     2:0.3               # ==> [0.0 0.3 0.0 0.0]
     2:0.8 3:-1.2 4:2.0  # ==> [0.0 0.8 -1.2 2.0]


Finally, note that the model topology specification should specify that the inputs are dense
and not sparse.  See the [configuration](Configuration).

## Vector Construction with an Alphabet

This input method does not require that the feature names correspond to integers referring
to the feature indices.  Instead, features names may be arbitrary strings. The input 
reader will build up a symbol table that maps each feature name to a unique integer, sometimes
referred to as an [Alphabet](../api/index.html#org.mitre.mandolin.util.Alphabet).

Whether the input vectors will be dense or sparse is determined in this case by the
model specification.  For a dense input representation, the input layer specification should
have the form: `{"ltype": "Input", ... }`.

To specify that the input vectors should be sparse, the input layer specification should appear
as: `{"ltype": "InputSparse",...}`.

See the [configuration](Configuration)

## Hashed Features


For datasets with a very large number of features, it can be much more efficient
to simply hash feature names in a way that maps each feature name to an integer
between `0` and `n`, where `n` is a specified constant. A potential downside
to this approach is that some feature names may hash to the same integer. If
this occurs infrequently enough, however, the impact on the model is minimal.
The integer `i` is obtained for a feature name `<feature_i>` represented
as a `String` via the expression:

    i = MurmurHash(<feature_i>) % n;

Where `n` is the specified size of the hash space and `%` is the modulo operator. The function
`MurmurHash` is the MurmurHash3 algorithm.

As the input vector space will be large and instances here will be sparse
the input layer specification should have the form: `{"ltype": "InputSparse",...}`


## Label File

For *all* input datasets the labels to be used by the classifier must be specified
in a separate file, indicated with the configuration option `mandolin.app.trainer.label-file`.
This file should contain the labels (as strings) that match the labels in 
`mandolin.app.trainer.train-file` and `mandolin.app.trainer.test-file`

