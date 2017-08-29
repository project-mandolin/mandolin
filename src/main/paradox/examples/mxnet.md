# MXNet

MXNet can be used in a variety of ways within Mandolin as demonstrated through the following
examples. Note that these examples all use the `mandolin-mx-0.3.6.jar` artifact and require that
the MXNet shared object has been compiled on the target platform.

## MNIST

    cd examples/mx_mnist
    java -Djava.library.path=<path to MXNet shared object> -Xmx2g -cp $MANDOLIN/dist/mandolin-mx-0.3.6.jar \
        org.mitre.mandolin.app.Mandolin --conf mx.conf

Note that this uses the same input files as the MMLP MNIST @ref[example](mmlp.md).
The config file specifies the mode as `train`. 

## MNIST via Native Format

Example using MNIST in a binary image format (not RecordIO).

## CalTech Image Dataset

Training image classification models is something deep neural networks excel at. To run this
example, the CalTech image dataset must be downloaded.

