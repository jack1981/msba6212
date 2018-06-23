package com.ssqcyy.msba6212.mlib

import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.apache.spark.sql.functions._
import com.ssqcyy.msba6212.evaluate.Evaluation
import org.apache.log4j.{ Level, Logger }
import com.ssqcyy.msba6212.utils.Utils
import com.ssqcyy.msba6212.utils.Utils.AppParams
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.optim.OptimMethod
import com.intel.analytics.bigdl.optim.Adam
import com.intel.analytics.bigdl.dlframes.DLEstimator
import com.intel.analytics.bigdl.dlframes.DLModel
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.sql.{ DataFrame, SQLContext, SaveMode }

object TrainDLPipeline {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("com.intel.analytics.bigdl").setLevel(Level.INFO)
    val st = System.nanoTime()

    Utils.trainParser.parse(args, Utils.AppParams()).foreach { param =>
      val conf = Engine.createSparkConf().setAppName("TrainWithNCF")
        .set("spark.sql.shuffle.partitions", param.defaultPartition.toString())
        .set("spark.sql.crossJoin.enabled", "true")
        .set("spark.sql.autobroadcastjointhreshold", "500000000")
      val sc = new SparkContext(conf)
      val spark = new SQLContext(sc)
      println("AppParams are " + param)
      val trainingStart = param.trainingStart
      val trainingEnd = param.trainingEnd
      val validationStart = trainingEnd
      val validationEnd = param.validationEnd
      val batchSize = param.batchSize
      val maxEpoch = param.maxEpoch
      val uOutput = param.uOutput
      val mOutput = param.mOutput
      val learningRate = param.learningRate
      val learningRateDecay = param.learningRateDecay
      DataPipeline.randomSampling = param.randomSampling
      DataPipeline.negRate = param.negRate
      DataPipeline.debug = param.debug
      Engine.init

      val dataDF = DataPipeline.loadPublicCSV(spark, param)
      val ulimit = dataDF.groupBy("uid").count().count().toInt
      val mlimit = dataDF.groupBy("mid").count().count().toInt
      val filterTrainingRawDF = dataDF
        .filter(s"date>=$trainingStart")
        .filter(s"date<=$trainingEnd")
        .drop("date").cache()

      val positiveTrainingDF = filterTrainingRawDF.select("uid", "mid").distinct().withColumn("label", lit(1.0f)).cache()
      val trainingDF = DataPipeline.mixNegativeAndCombineFeatures(positiveTrainingDF, filterTrainingRawDF, param, true)
      val featureTrainDF = DataPipeline.norm(trainingDF)
      val trainingCount = featureTrainDF.count()

      println("Start Deep Learning trainning , training records count: " + trainingCount + " batchSize is " + batchSize + " maxEpoch is " + maxEpoch + " learningRate is " + learningRate + " learningRateDecay is " + learningRateDecay)
      val model = getModel(ulimit, uOutput, mlimit, mOutput, param, 2)

      val criterion = ClassNLLCriterion()
      val dlc = new DLEstimator(model, criterion, Array(2), Array(1))
        .setBatchSize(batchSize)
        .setOptimMethod(new Adam())
        .setLearningRate(learningRate)
        .setLearningRateDecay(learningRateDecay)
        .setMaxEpoch(maxEpoch)

      val dlModel = dlc.fit(featureTrainDF)

      val filterValidationRawDF = dataDF
        .filter(s"date>$validationStart")
        .filter(s"date<=$validationEnd")
        .drop("date").cache()

      val positiveValidationDF = filterValidationRawDF.select("uid", "mid").distinct().withColumn("label", lit(1.0f)).cache()

      val validationDF = DataPipeline.mixNegativeAndCombineFeatures(positiveValidationDF, filterValidationRawDF, param, true)

      val featureValidationDF = DataPipeline.norm(validationDF)

      evaluate(dlModel, featureValidationDF, param);

      println("total time: " + (System.nanoTime() - st) / 1e9)
      sc.stop()
    }

  }

  // uid, mid, count
  private[ssqcyy] def evaluate(model: DLModel[Float], validDF: DataFrame, param: AppParams): Unit = {

    println("positiveDF count: " + validDF.count())
    println("validationDF count: " + validDF.count())

    val label2Binary = udf { d: Float => if (d == 2.0f) 0.0 else 1.0 }
    val raw2Classification = udf { d: Seq[Double] =>
      require(d.length == 2, "actual length:" + d.length)
      if (d(0) > d(1)) 1.0
      else 0.0
    }

    val deepPredictions = model.transform(validDF)
    val evaluateDF = deepPredictions
      .withColumnRenamed("prediction", "raw")
      .withColumn("label", label2Binary(col("label")))
      .withColumn("prediction", raw2Classification(col("raw")))

    Evaluation.evaluate(evaluateDF)

  }

  private[ssqcyy] def getModel(
    ulimit: Int, uOutput: Int,
    mlimit: Int, mOuput: Int,
    param: AppParams,
    featureDimension: Int): Module[Float] = {

    val initialWeight = 0.5

    val user_table = LookupTable(ulimit, uOutput)
    val item_table = LookupTable(mlimit, mOuput)

    user_table.setWeightsBias(Array(Tensor[Float](ulimit, uOutput).randn(0, initialWeight)))
    item_table.setWeightsBias(Array(Tensor[Float](mlimit, mOuput).randn(0, initialWeight)))

    val numExtraFeature = featureDimension - 2
    val embedded_layer = if (numExtraFeature > 0) {
      Concat(2)
        .add(Sequential().add(Select(2, 1)).add(user_table))
        .add(Sequential().add(Select(2, 2)).add(item_table))
        .add(Sequential().add(Narrow(2, 3, numExtraFeature)))
    } else {
      Concat(2)
        .add(Sequential().add(Select(2, 1)).add(user_table))
        .add(Sequential().add(Select(2, 2)).add(item_table))
    }

    val model = Sequential()
    model.add(embedded_layer)

    val numEmbeddingOutput = uOutput + mOuput + numExtraFeature
    val linear1 = math.pow(2.0, (math.log(numEmbeddingOutput) / math.log(2)).toInt).toInt
    val linear2 = linear1 / 2
    model.add(Linear(numEmbeddingOutput, linear1)).add(ReLU()).add(Dropout())
    model.add(Linear(linear1, linear2)).add(ReLU())
    model.add(Linear(linear2, 2)).add(LogSoftMax())
    if (param.debug) {
      println(model)
    }
    model
  }

}
