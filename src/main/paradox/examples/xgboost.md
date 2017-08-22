# XGBoost

Mandolin provides a wrapper for XGBoost.  The standard Mandolin file format is used (rather than XGBoost's native
format). The wrapper also facilitates use of model selection over XGBoost hyper-parameters. (see . . . )

## Splice Example

A small excerpt of the full Splice dataset is used here to demonstrate the utiltiy of XGBoost
for modeling problems best captured with a non-linear classifier. The example can be run via:

    cd $MANDOLIN/examples/splice
    java -Djava.library.path=$MANDOLIN/mandolin-mx/pre-compiled/xgboost/linux -Xmx2g -cp ../../dist/mandolin-mx-0.3.5.jar --conf splice.conf



