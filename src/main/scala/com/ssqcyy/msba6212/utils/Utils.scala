package com.ssqcyy.msba6212.utils

import org.apache.spark.SparkContext
import scopt.OptionParser

object Utils {
  val seperator = ","
  case class AppParams(
    batchSize: Int =1000,
    maxEpoch: Int = 10,
    uOutput: Int = 200,
    trainingStart: Int = 20160101,
    trainingEnd: Int = 20161101,
    validationEnd: Int = 20161231,
    defaultPartition: Int = 60,
    mOutput: Int = 100,
    learningRate: Double = 1e-3,
    learningRateDecay: Double = 1e-7,
    rank: Int = 10,
    brank: Int = 50,
    regParam: Double = 0.01,
    bregParam: Double = 0.20,
    alpha: Double = 0.01,
    balpha: Double = 0.15,
    debug: Boolean = false,
    saveModel: Boolean = false,
    randomSampling: Boolean = false,
    negRate: Double = 1.0,
    dataFilePath: String = "/home/msba6212/data/pcard.csv",
    clusterParams: ClusterParams = ClusterParams(2, 30, 3))

  val trainParser = new OptionParser[AppParams]("MSBA 6212 FINAL PROJECT") {
    head("AppParams:")
    opt[Int]("batchSize")
      .text("batch size for deep learning")
      .action((x, c) => c.copy(batchSize = x))
    opt[Int]("maxEpoch")
      .text("max Epoch")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Int]("uOutput")
      .text("User matrix output")
      .action((x, c) => c.copy(uOutput = x))
    opt[Int]("trainingStart")
      .text("trainingStart time yyyymmdd")
      .action((x, c) => c.copy(trainingStart = x))
    opt[Int]("trainingEnd")
      .text("trainingEnd time yyyymmdd")
      .action((x, c) => c.copy(trainingEnd = x))
    opt[Int]("validationEnd")
      .text("validationEnd time yyyymmdd")
      .action((x, c) => c.copy(validationEnd = x))
    opt[Int]("defaultPartition")
      .text("default spark shuff Partition number")
      .action((x, c) => c.copy(defaultPartition = x))
    opt[Int]("mOutput")
      .text("Merchant matrix output")
      .action((x, c) => c.copy(mOutput = x))
    opt[Double]("learningRate")
      .text("Learning Rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("Learning Rate Decay")
      .action((x, c) => c.copy(learningRateDecay = x))
    opt[Int]("rank")
      .text("ALS rank")
      .action((x, c) => c.copy(rank = x))
    opt[Int]("brank")
      .text("ALS benchmark Rank")
      .action((x, c) => c.copy(brank = x))
    opt[Double]("regParam")
      .text("ALS regParam")
      .action((x, c) => c.copy(regParam = x))
    opt[Double]("bregParam")
      .text("ALS benchmark regParam")
      .action((x, c) => c.copy(bregParam = x))
    opt[Double]("alpha")
      .text("ALS alpha")
      .action((x, c) => c.copy(alpha = x))
    opt[Double]("balpha")
      .text("ALS benchmark alpha")
      .action((x, c) => c.copy(balpha = x))
    opt[Boolean]("debug")
      .text("turn on debug mode or not")
      .action((x, c) => c.copy(debug = x))
    opt[Boolean]("randomSampling")
      .text("force to use random Sampling")
      .action((x, c) => c.copy(randomSampling = x))
    opt[Double]("negRate")
      .text("negRate")
      .action((x, c) => c.copy(negRate = x))
    opt[String]("dataFilePath")
      .text("dataFilePath")
      .action((x, c) => c.copy(dataFilePath = x))
    opt[String]("clusterParams")
      .text("ClusterParams")
      .action((x, c) => {
        val pArr = x.split(seperator).map(_.trim)
        val p = ClusterParams(pArr(0).toInt, pArr(1).toInt, pArr(2).toInt)
        c.copy(clusterParams = p)
      })
  }
  case class ClusterParams(
    numClusters: Int,
    numIterations: Int,
    runTimes: Int)
}
