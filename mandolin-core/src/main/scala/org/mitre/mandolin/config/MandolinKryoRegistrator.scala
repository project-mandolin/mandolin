package org.mitre.mandolin.config
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

import com.esotericsoftware.kryo.Kryo
import com.twitter.chill.AllScalaRegistrar

trait MandolinRegistrator {
  def register(kryo: Kryo) = {
    (new AllScalaRegistrar).apply(kryo)
    kryo.register(classOf[collection.mutable.HashMap[Int,Double]])
    kryo.register(classOf[cern.colt.map.OpenIntDoubleHashMap])
    kryo.register(classOf[org.mitre.mandolin.util.Tensor1])
    kryo.register(classOf[org.mitre.mandolin.util.Tensor2])
    kryo.register(classOf[org.mitre.mandolin.util.DenseTensor1])
    kryo.register(classOf[org.mitre.mandolin.util.SparseTensor1])
    kryo.register(classOf[org.mitre.mandolin.util.DenseTensor2])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPModelSpec])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPFactor])
    kryo.register(classOf[org.mitre.mandolin.mlp.MandolinMLPSettings])
    kryo.register(classOf[org.mitre.mandolin.mselect.ModelSelectionSettings])
    kryo.register(classOf[org.mitre.mandolin.mselect.ModelSpace])
    kryo.register(classOf[org.mitre.mandolin.mlp.StdMMLPFactor])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPWeights])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPLayout])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPLossGradient])
    kryo.register(classOf[org.mitre.mandolin.mlp.SparseToDenseReader])
    kryo.register(classOf[org.mitre.mandolin.mlp.DenseReader])
    kryo.register(classOf[org.mitre.mandolin.mlp.MMLPInstanceEvaluator[_]])
    kryo.register(classOf[org.mitre.mandolin.mlp.VecFeatureExtractor])
    kryo.register(classOf[org.mitre.mandolin.mlp.Layer])
    kryo.register(classOf[org.mitre.mandolin.mlp.DenseInputLayer])
    kryo.register(classOf[org.mitre.mandolin.mlp.SparseInputLayer])
    kryo.register(classOf[org.mitre.mandolin.mlp.LayerDesignate])
    kryo.register(classOf[org.mitre.mandolin.mlp.ANNetwork])
    kryo.register(classOf[org.mitre.mandolin.mlp.LType])
    kryo.register(classOf[org.mitre.mandolin.util.Alphabet])
    kryo.register(classOf[org.mitre.mandolin.util.StdAlphabet])
    kryo.register(classOf[org.mitre.mandolin.util.RandomAlphabet])
    kryo.register(classOf[org.mitre.mandolin.util.IdentityAlphabet])
    kryo.register(classOf[org.mitre.mandolin.embed.EmbedWeights])
    kryo.register(classOf[org.mitre.mandolin.embed.NullUpdater])
    kryo.register(classOf[org.mitre.mandolin.embed.EmbedAdaGradUpdater])
    kryo.register(classOf[org.mitre.mandolin.embed.SeqInstance])
    kryo.register(classOf[org.mitre.mandolin.embed.SeqInstanceExtractor])
  }
}
