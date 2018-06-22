package com.ssqcyy.msba6212.mlib

import org.apache.spark.ml.recommendation.{ ALS, ALSModel }
import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{ DataFrame, SparkSession }
import org.apache.spark.sql.functions._
import com.ssqcyy.msba6212.evaluate.Evaluation
import org.apache.log4j.{ Level, Logger }
import com.ssqcyy.msba6212.utils.Utils
import com.ssqcyy.msba6212.utils.Utils.AppParams
import org.apache.hadoop.fs.Path

object TrainPipeline {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
     val st = System.nanoTime()

    Utils.trainParser.parse(args, Utils.AppParams()).foreach { param =>
      val spark = SparkSession.builder().appName("TrainWithALS")
        .config("spark.sql.shuffle.partitions", param.defaultPartition).getOrCreate()
      println("AppParams are " + param)
      val trainingStart = param.trainingStart
      val trainingEnd = param.trainingEnd
      val validationStart = trainingEnd
      val validationEnd = param.validationEnd
      val maxEpoch = param.maxEpoch
      val rank = param.uOutput
      DataPipeline.randomSampling = param.randomSampling
      DataPipeline.negRate = param.negRate
      DataPipeline.debug = param.debug
      val numClusters = param.clusterParams.numClusters
      val numIterations = param.clusterParams.numIterations
      val runTimes = param.clusterParams.runTimes
      import spark.implicits._

      val dataDF = DataPipeline.loadPublicCSV(spark,param)
      val filterTrainingRawDF = dataDF
        .filter(s"date>=$trainingStart")
        .filter(s"date<=$trainingEnd")
        .drop("date").cache()

      val trainingRawDF = filterTrainingRawDF.groupBy("uid", "mid").count().withColumnRenamed("count", "label").cache()
      val trainingDF = DataPipeline.mixNegativeAndCombineFeatures(trainingRawDF, filterTrainingRawDF,param)
      val trainingCount = trainingDF.count()

      val clusterTrainDF = DataPipeline.normFeatures(trainingDF,param)

      println("Start Kmeans trainning , training records count: " + trainingCount + " numClusters is " + numClusters + " numIterations is " + numIterations + " runTimes is " + runTimes)

      val idTrainFeaturesRDD = clusterTrainDF.rdd.map(s => (s.getDouble(0), Vectors.dense(s.getSeq[Float](3).toArray.map { x => x.asInstanceOf[Double] }))).cache()
      // Cluster the data into two classes using KMeans
      val clusters: KMeansModel = KMeans.train(idTrainFeaturesRDD.map(_._2), numClusters, numIterations, runTimes, KMeans.RANDOM)
      println("Cluster Number:" + clusters.clusterCenters.length)
      println("Cluster Centers Information Overview:")
      var clusterIndex: Int = 0
      clusters.clusterCenters.foreach(
        x => {
          println("Center Point of Cluster " + clusterIndex + ":")
          println(x)
          clusterIndex += 1
        })

      val filterValidationRawDF= dataDF
        .filter(s"date>$validationStart")
        .filter(s"date<=$validationEnd")
        .drop("date").cache()
        
      val validationRawDF = filterValidationRawDF.groupBy("uid", "mid").count().withColumnRenamed("count", "label").cache()

      val validationDF = DataPipeline.mixNegativeAndCombineFeatures(validationRawDF,filterValidationRawDF, param)

      val clusterValidationDF = DataPipeline.normFeatures(validationDF,param)

      val idValidationFeaturesRDD = clusterValidationDF.rdd.map(s => (s.getDouble(0), Vectors.dense(s.getSeq[Float](3).toArray.map { x => x.asInstanceOf[Double] }))).cache()
      val clustersRDD = clusters.predict(idValidationFeaturesRDD.map(_._2))
      val idClusterRDD = idValidationFeaturesRDD.map(_._1).zip(clustersRDD)
      val kMeanPredictions = idClusterRDD.toDF("uid", "cluster")
      val clusterPredictionsDF = kMeanPredictions
        .select("uid", "cluster")
        .cache()

      var clusterNumber: Int = 0
      while (clusterNumber < numClusters) {
        println("Start ALS pipeline for cluster: " + clusterNumber);
        val clusterSingleDF = clusterPredictionsDF.filter(col("cluster") === clusterNumber).distinct();
        println("Count of cluster: " + clusterNumber + " is " + clusterSingleDF.count());
        val trainingClusterDF = trainingDF.join(clusterSingleDF, Array("uid"), "inner");
        val validationClusterDF = validationDF.join(clusterSingleDF, Array("uid"), "inner");
        println("Split data into Training and Validation for cluster : "+ clusterNumber + ":");
        println("cluster : "+ clusterNumber + ":"+" training records count: " + trainingClusterDF.count());
        println("cluster : "+ clusterNumber + ":"+" validation records count: " + validationClusterDF.count());

        val als = new ALS()
          .setMaxIter(maxEpoch)
          .setRegParam(0.01)
          .setRank(rank)
          .setAlpha(0.01)
          .setUserCol("uid")
          .setItemCol("mid")
          .setRatingCol("label");
        val model = als.fit(trainingClusterDF);

        evaluate(model, validationClusterDF, param);
        println("cluster : "+ clusterNumber + ":"+"Train and Evaluate End");
        clusterNumber += 1
      }
      
      
      println("total time: " + (System.nanoTime() - st) / 1e9)
      spark.stop()
    }

  }

  // uid, mid, count
  private[ssqcyy] def evaluate(model: ALSModel, validDF: DataFrame, param: AppParams): Unit = {

    println("positiveDF count: " + validDF.count())
    println("validationDF count: " + validDF.count())
    val prediction = model.transform(validDF)
    val label2Binary = udf { d: Float => if (d == 2.0f) 0.0 else 1.0 }
    val prediction2Binary = udf { d: Double => if (d > 1.0) 1.0 else 0.0 }

    val evaluateDF = prediction
      .withColumn("label", label2Binary(col("label")))
      .withColumn("prediction", prediction2Binary(col("prediction")))
    Evaluation.evaluate(evaluateDF)

  }

}
