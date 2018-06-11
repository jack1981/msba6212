package com.ssqcyy.msba6212.evaluate

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.functions._


object Evaluation {

  /**
    *
    * userid, mid, label, prediction
   */
  def evaluate(evaluateDF: DataFrame): Unit = {
    val topN = 20
    evaluateDF.cache()

    evaluateDF.orderBy(rand()).show(20, false)
    val binaryEva = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
    println("AUROC: " + binaryEva.evaluate(evaluateDF))

    val binaryEva2 = new BinaryClassificationEvaluator().setRawPredictionCol("prediction")
      .setMetricName("areaUnderPR")
    println("AUPRCs: " + binaryEva2.evaluate(evaluateDF))

    val tp = evaluateDF.filter("prediction=1").filter("label=1").count()
    val fp = evaluateDF.filter("prediction=1").filter("label=0").count()
    val fn = evaluateDF.filter("prediction=0").filter("label=1").count()

    println(s"tp: $tp")
    println(s"fp: $fp")
    println(s"fn: $fn")

    println("recall: " + tp.toDouble / (tp + fn))
    println("precision: " + tp.toDouble / (tp + fp))

    println("label distribution: ")
    evaluateDF.groupBy("label").count().show()

    println("prediction distribution: ")
    evaluateDF.groupBy("prediction").count().show()

    if(evaluateDF.columns.contains("prob")) {
      val topNStat = evaluateDF.select("features", "label", "prob", "prediction").filter("prediction=1")
        .rdd.map { case Row(feature: Seq[Float], label: Double, prob: Double, prediction: Double) =>
        val uid = feature(0).toInt
        val mid = feature(1).toInt
        (uid, mid, label, prob, prediction)
      }.groupBy(_._1).map { case (uid, iter) =>
        val records = iter.toArray
        val topNprob = records.sortBy(-_._4).take(topN).filter(_._5 == 1)
        val correctTopNprob = topNprob.count(_._3 == 1)
        (correctTopNprob, topNprob.length)
      }.reduce((t1, t2) => (t1._1 + t2._1, t1._2 + t2._2))
      val topNprecision = topNStat._1.toDouble / topNStat._2

      println(s"$topN precision: $topNprecision")
    }

    require(evaluateDF.select(col("label").isin(0.0f, 1.0f)).rdd
      .map(r => r.getBoolean(0)).collect().forall(t => t), "label should all in 0, 1")

    require(evaluateDF.select(col("prediction").isin(0.0, 1.0)).rdd
      .map(r => r.getBoolean(0)).collect().forall(t => t), "label should all in 0, 1")
    evaluateDF.unpersist()
  }



}
