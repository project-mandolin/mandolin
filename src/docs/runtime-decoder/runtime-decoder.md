{%
  title: Runtime Decoder
%}

# Runtime Decoder

To use a trained model as part of a larger application, it's helpful to have a simple runtime *decoder*
that can be used to process new data instances and produce a prediction and/or generate a full posterior
distribution.  Mandolin provides a simple mechanism to do this allowing a model to be called easily from
Scala or Java.  From Scala, a simple case might be:

    import org.mitre.mandolin.app.GlpRuntimeDecoder
    import org.mitre.mandolin.app.StringDoublePair

    val modelFile = new java.io.File(modelFilePath)
    val decoder = new GlpRuntimeDecoder(modelFile)
    val fv = List(StringDoublePair("1",-0.722222), 
                  StringDoublePair("2",0.166667), 
                  StringDoublePair("3",-0.694915),
                  StringDoublePair("4",-0.916667))

    // run the model against the feature vector, fv
    // get the most probable explanation along with its posterior probability
    val mpe : StringDoublePair = decoder.decodePairsMPEWithScore(fv)

The same functionality is realized in Java with:

    import org.mitre.mandolin.app.GlpRuntimeDecoder;
    import org.mitre.mandolin.app.StringDoublePair;
    import java.util.List;
    import java.util.ArrayList;

    File modelFile = new File(modelFilePath);
    GlpRuntimeDecoder decoder = new GlpRuntimeDecoder(modelFile);
    List<StringDoublePair> fv = new ArrayList<StringDoublePair>();
    fv.add(new StringDoublePair("1",-0.722222),
           new StringDoublePair("2",0.166667), 
           new StringDoublePair("3",-0.694915),
           new StringDoublePair("4",-0.916667));

    StringDoublePair mpe = decoder.decodePairsMPEWithScore(fv);

Note that the `GlpRuntimeDecoder` class includes methods for handling
both Scala and Java lists.

