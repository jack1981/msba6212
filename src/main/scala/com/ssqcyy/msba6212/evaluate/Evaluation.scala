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

    evaluateDF.unpersist()
  }



}
