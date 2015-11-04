package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar
import org.apache.spark.serializer.KryoRegistrator

/**
 * Provides and registers a set of classes that will be serialized/deserialized
 * using Kryo for use within Spark.
 * @author wellner
 */
class MandolinKryoRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) = {
    (new AllScalaRegistrar).apply(kryo)
    kryo.register(classOf[collection.mutable.HashMap[Int,Double]])
    kryo.register(classOf[cern.colt.map.OpenIntDoubleHashMap])
    kryo.register(classOf[org.apache.spark.util.Vector])
    kryo.register(classOf[org.mitre.mandolin.util.Tensor1])
    kryo.register(classOf[org.mitre.mandolin.util.Tensor2])
    kryo.register(classOf[org.mitre.mandolin.util.DenseTensor1])
    kryo.register(classOf[org.mitre.mandolin.util.SparseTensor1])
    kryo.register(classOf[org.mitre.mandolin.util.DenseTensor2])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPModelSpec])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPFactor])
    kryo.register(classOf[org.mitre.mandolin.glp.StdGLPFactor])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPWeights])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPLayout])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPLossGradient])
    kryo.register(classOf[org.mitre.mandolin.glp.SparseToDenseReader])
    kryo.register(classOf[org.mitre.mandolin.glp.DenseReader])
    kryo.register(classOf[org.mitre.mandolin.glp.GLPInstanceEvaluator])
    kryo.register(classOf[org.mitre.mandolin.glp.VecFeatureExtractor])
    kryo.register(classOf[org.mitre.mandolin.glp.Layer])
    kryo.register(classOf[org.mitre.mandolin.glp.DenseInputLayer])
    kryo.register(classOf[org.mitre.mandolin.glp.LayerDesignate])
    kryo.register(classOf[org.mitre.mandolin.glp.ANNetwork])
    kryo.register(classOf[org.mitre.mandolin.glp.LType])
    kryo.register(classOf[org.mitre.mandolin.util.Alphabet])
    kryo.register(classOf[org.mitre.mandolin.util.StdAlphabet])
    kryo.register(classOf[org.mitre.mandolin.util.RandomAlphabet])
    kryo.register(classOf[org.mitre.mandolin.util.IdentityAlphabet])
  }
}
