#!/bin/bash
spark-submit \
--master local \
--deploy-mode client \
--class com.ssqcyy.msba6212.mlib.TrainPipeline \
--executor-memory 2G \
--driver-memory 1g \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.ui.showConsoleProgress=false \
--conf spark.yarn.max.executor.failures=4 \
--conf spark.yarn.executor.memoryOverhead=512 \
--conf spark.yarn.driver.memoryOverhead=512 \
--conf spark.sql.tungsten.enabled=true \
--conf spark.locality.wait=1s \
--conf spark.yarn.maxAppAttempts=4 \
--conf spark.serializer=org.apache.spark.serializer.JavaSerializer \
ml-6212.jar \
--trainingStart 20130530 \
--trainingEnd 20140615 \
--validationEnd 20140630 \
--uOutput 100 \
--maxEpoch 25 \
--defaultPartition 10 \
--dataFilePath "/home/msba6212/data/pcard.csv" \
--negRate 0.2 \
--randomSampling true \
--debug true