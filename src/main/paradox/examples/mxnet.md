# MXNet

MXNet can be used in a variety of ways within Mandolin as demonstrated through the following
examples. Note that these examples all use the `mandolin-mx-0.3.6.jar` artifact and require that
the MXNet shared object has been compiled on the target platform. Assume for these examples,
the necessary object file(s) are located in `$HOME/lib`. Assume that $MXNET defines the home
directory for MXNet and that $MANDOLIN defines the location of Mandolin.

## MNIST

    cd $MANDOLIN/examples/mx_mnist
    java -Djava.library.path=$HOME/lib -Xmx2g -cp $MANDOLIN/dist/mandolin-mx-0.3.6.jar \
        org.mitre.mandolin.app.Mandolin --conf mx.conf

Note that this uses the same input files as the MMLP MNIST @ref[example](mmlp.md).
The config file specifies the mode as `train`. 

## MNIST via Native Format

Example using MNIST in a binary image format (not RecordIO).

## CalTech Image Dataset

Training image classification models is the canonical deep neural networks task. This example involves
training and evaluating the model on a real (but small) image dataset, the CalTech 101 dataset.

### Download the Data

First, grab the image data which can be downloaded via the Download at:

    http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

Create a space for the data:

    mkdir $MANDOLIN/examples/caltech/data
    mv 101_ObjectCategories.tar.gz $MANDOLIN/examples/caltech/data

Unpack the data into the `$MANDOLIN/examples/caltech/data` directory:

    cd $MANDOLIN/examples/caltech/data
    tar xzvf 101_ObjectCategories.tar.gz

### Build the RecordIO Files

The next step is to build the RecordIO file that contains the labeled images in a packed format. This
can be done via MXNet's `im2rec.py` script.  Assuming MXNet has been built and installed properly, from
`$MANDOLIN/examples/caltech`, run:

    python $MXNET/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 --train-ratio=0.8 --test-ratio=0.2 data/caltech data/101_ObjectCategories

This will create the files `data/caltech_train.lst` and `data/caltech_test.lst` which are listings of the image files to be
included in the training and test sets, respectively.

The same script must be invoked again to build the RecordIO files for the training and test sets:

    python $MXNET/tools/im2rec.py --num-thread=12 --pass-through=1  data/caltech data/101_ObjectCategories

The number of threads used to build the packed file(s) can be adjusted as appropriate for the machine it's run on.  After finishing,
the files `data/caltech_train.rec` and `data/caltech_test.rec` should be found in the `$MANDOLIN/examples/caltech/data/` directory.

### Train and Evaluate a Model

The final step involves invoking Mandolin to build a model via MXNet.  This assumes that MXNet has been built on the target
platform and the necesary shared object files reside in an appropriate `lib` directory.  

Mandolin is invoked on this example as:

    java -Djava.library.path=$HOME/lib -Xmx6g -cp $MANDOLIN/dist/mandolin-mx-0.3.6.jar org.mitre.mandolin.app.Mandolin --conf caltech.conf

As usual, the various training parameters are specified in the configuration file. For this example, a fairly simple (and arbitrary)
convolutional neural network is used as the architecture.  More complex architectures, details of which can be found in the
documentation section on MXNet Specifications.
