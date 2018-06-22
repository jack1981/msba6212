# MSBA 6212 
MSBA 6212 is the outcomes for final project belongs to MSBA 6212 course.
  - In this project , I am going to do a simple POC about User-Item (Item type likes Merchant, Category,Geo...) Propensity Models by leveraging multiple data mining models and technologies connecting real business data sets and problems
  - The output User-Item Propensity Models from this project which is for Marketing draw from transaction behavior to provide objective information about likely future behaviors in the areas of card use, likely preference and retention

# Proposal 
  - I designed two steps of work for this project : 
   1) Pre-compute the clusters (segmentation) of users. Considering the amount of users and items are both large and it is hard to calculate all combinations of user-item propensity model, so cluster users based on behavior similarities is a very popular way to divided big scopes to small parts in parallel. For clustering of users , I am going to try Kmeans non-supervisor learning model.
   2) Based on the clusters , abstract the training/test data for each cluster and fit collaborative filtering model and predict user-item propensity scores
   I am going to try Spark Mlib (https://spark.apache.org/mllib/) as machine learning library and Apache Spark is the run time container to run machine learning job and the programming language will be Scala
   I am going to use real credit transactions from public data source (https://catalog.data.gov/dataset/purchase-card-pcard-fiscal-year-2014) and will run the machine learning job at Spark cluster at PROD. But I also will enable the project run-able artifacts at single VM with a single instance of Spark with simulate data sets for final project code review.

# Installation
MSBA 6212 requires docker container to run if you are working at a windows pc or laptop ( prefer windows 10) 
## Install Docker and Docker Toolbox
### Install the docker for window
https://store.docker.com/editions/community/docker-ce-desktop-windows
- please use this docker account
username:msba6212
password:hadoop123
- Get Docker CE for Windows (stable)
- Double-click Docker for Windows Installer to run the installer.
- When the installation finishes, Docker starts automatically. The whale  in the notification area indicates that Docker is running, and accessible from a terminal.
### Install the Docker Toolbox
- https://docs.docker.com/toolbox/toolbox_install_windows/
- After installation , click Kitematic (Alpha) shortcut
- Then click DOCKER-CLI on the left corner, you will enter a docker cli window

# Create a standalone Spark environment
MSBA 6212 requires a standalone spark envrionment to run Spark Mllib jobs
## build a docker image
create a msba6212 folder at c:\ and download https://github.com/jack1981/msba6212/blob/master/docker/spark.df
and https://github.com/jack1981/msba6212/blob/master/docker/docker-compose.yml two files under this folder
```sh
$ cd C:\msba6212\
$ docker build -f spark.df -t spark .
$ docker images
```
You should see 
```sh
$ docker images
REPOSITORY                 TAG                 IMAGE ID            CREATED             SIZE
spark                      latest              c387a9fa5ef3        8 seconds ago       923MB
ubuntu                     16.04               5e8b97a2a082        6 days ago          114MB
```
## start the standalone spark container
```sh
$ $env:COMPOSE_CONVERT_WINDOWS_PATHS=1
$ docker run -it -p 8088:8088 -p 8042:8042 -p 4041:4040 --name driver -h driver spark:latest bash
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7#
```
**Note! don't close window or exit the shell , then the container will be terminated , if you want to quit the container and want to attach it back , you should Ctrl+p then Ctrl+q to leave container safely**

## prepare tools at the container 
MSBA 6212 requires those tools , vim, maven, git and unzip
```sh
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# apt-get update
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# apt-get install vim
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# apt-get install maven
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# apt-get install git
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# apt-get install unzip
```
# Build the artifacts
I created a github project for MSBA 6212 and need to download artifacts , build and run
```sh
root@driver:/usr/local/spark-2.2.0-bin-hadoop2.7# cd /home
root@driver:/home# git clone https://github.com/jack1981/msba6212.git
```
Build the project with maven command
```sh
root@driver:/home# cd msba6212/
root@driver:/home/msba6212# mvn clean install
```
Unzip the data file 
```sh
root@driver:/home/msba6212# cd data
root@driver:/home/msba6212/data# unzip pcard.zip
```
copy the rename the jar
```sh
root@driver:/home/msba6212/data# cd ..
root@driver:/home/msba6212# mv target/ml-6212-0.1-SNAPSHOT-jar-with-dependencies.jar ml-6212.jar
```

# Run the project and check the results
## explain the parameters 
```sh
--trainingStart 20130530 # training start date 
--trainingEnd 20140615  # training end date
--validationEnd 20140630 # validation end date 
--rank 10 # value of ALS rank parameter 
--brank 50 # value of benchmark rank parameter 
--regParam 0.01 # value of ALS regParam parameter 
--bregParam 0.20 # value of benchmark regParam parameter 
--alpha 0.01 # value of ALS alpha parameter 
--balpha 0.15 # value of benchmark alpha parameter
--maxEpoch 10 # value of ALS max iterations parameter 
--defaultPartition 10 # spark shuffling partition
--dataFilePath "/home/msba6212/data/pcard.csv" # the path of data source csv
--negRate 0.2 # the rate to generate negtive sampling 
--randomSampling true # Sampling mode
--debug true # turn on debug or not 
```
## execute the run script
```sh
root@driver:/home/msba6212# ./run_als.sh
```
## the expected log
```sh
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
18/06/22 18:56:42 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
AppParams are AppParams(10,200,20130530,20140615,20140630,10,100,0.001,1.0E-7,10,50,0.01,0.2,0.01,0.15,true,false,true,0.2,/home/msba6212/data/pcard.csv,ClusterParams(5,30,3))
positive samples count: 67334
ulimit count: 5178
mlimit count: 435
randomNegativeSamples
combinedDF count: 80361
18/06/22 18:57:59 WARN Executor: Managed memory leak detected; size = 17039360 bytes, TID = 252
+------+-----+-----+-----------+-----------+
|   uid|  mid|label|totalVisits|totalAmount|
+------+-----+-----+-----------+-----------+
|1717.0|342.0|  0.0|          0|        0.0|
| 859.0|403.0|  0.0|          0|        0.0|
|3862.0|278.0|  0.0|          0|        0.0|
|2872.0|112.0|  0.0|          0|        0.0|
|2260.0|392.0|  0.0|          0|        0.0|
+------+-----+-----+-----------+-----------+
only showing top 5 rows

original features:
18/06/22 18:58:07 WARN Executor: Managed memory leak detected; size = 17039360 bytes, TID = 386
+------+-----+-----+-----------+-----------+
|   uid|  mid|label|totalVisits|totalAmount|
+------+-----+-----+-----------+-----------+
|1717.0|342.0|  0.0|          0|        0.0|
| 859.0|403.0|  0.0|          0|        0.0|
|3862.0|278.0|  0.0|          0|        0.0|
|2872.0|112.0|  0.0|          0|        0.0|
|2260.0|392.0|  0.0|          0|        0.0|
+------+-----+-----+-----------+-----------+
only showing top 5 rows

featureCols are CAST(totalVisits AS DOUBLE),CAST(totalAmount AS DOUBLE)
Start Kmeans trainning , training records count: 80361 numClusters is 5 numIterations is 30 runTimes is 3
18/06/22 18:58:14 WARN KMeans: The input data is not directly cached, which may hurt performance if its parent RDDs are also uncached.
18/06/22 18:58:18 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
18/06/22 18:58:18 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
18/06/22 18:58:27 WARN KMeans: The input data was not directly cached, which may hurt performance if its parent RDDs are also uncached.
Cluster Number:5
Cluster Centers Information Overview:
Center Point of Cluster 0:
[1156.738076120122,71.39173491783096,-0.02212103008614722,-0.021208758409307197]
Center Point of Cluster 1:
[333.59378661172383,72.26485678503616,0.15934356683714504,0.0992276658875982]
Center Point of Cluster 2:
[3122.7409138528487,83.72721411878474,-0.10112104972586763,-0.06441488978332281]
Center Point of Cluster 3:
[2077.924583492306,75.42369324685235,-0.06916197859695172,-0.048852429138457665]
Center Point of Cluster 4:
[4342.994540491356,116.86646951774341,-0.13518378782099633,-0.05600590130375378]
positive samples count: 8924
ulimit count: 2824
mlimit count: 260
randomNegativeSamples
combinedDF count: 10683
18/06/22 18:58:37 WARN Executor: Managed memory leak detected; size = 17039360 bytes, TID = 1350
+------+-----+-----+-----------+-----------+
|   uid|  mid|label|totalVisits|totalAmount|
+------+-----+-----+-----------+-----------+
| 864.0| 15.0|  0.0|          0|        0.0|
|2798.0| 83.0|  0.0|          0|        0.0|
|2507.0|154.0|  0.0|          0|        0.0|
| 102.0|107.0|  0.0|          0|        0.0|
| 851.0|149.0|  0.0|          0|        0.0|
+------+-----+-----+-----------+-----------+
only showing top 5 rows

original features:
18/06/22 18:58:38 WARN Executor: Managed memory leak detected; size = 17039360 bytes, TID = 1412
+------+-----+-----+-----------+-----------+
|   uid|  mid|label|totalVisits|totalAmount|
+------+-----+-----+-----------+-----------+
| 864.0| 15.0|  0.0|          0|        0.0|
|2798.0| 83.0|  0.0|          0|        0.0|
|2507.0|154.0|  0.0|          0|        0.0|
| 102.0|107.0|  0.0|          0|        0.0|
| 851.0|149.0|  0.0|          0|        0.0|
+------+-----+-----+-----------+-----------+
only showing top 5 rows

featureCols are CAST(totalVisits AS DOUBLE),CAST(totalAmount AS DOUBLE)
Start ALS pipeline for cluster: 0
Count of cluster: 0 is 763
Split data into Training and Validation for cluster : 0:
cluster : 0: training records count: 16307
cluster : 0: validation records count: 2603
18/06/22 18:59:00 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
18/06/22 18:59:00 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
18/06/22 19:02:51 WARN Executor: Managed memory leak detected; size = 17039360 bytes, TID = 9928
best rmse  = 15.647733318034643
best rank = 50
positiveDF count: 2603
validationDF count: 2603
+------+-----+-----+-----------+------------------+-------+----------+
|uid   |mid  |label|totalVisits|totalAmount       |cluster|prediction|
+------+-----+-----+-----------+------------------+-------+----------+
|1191.0|114.0|1.0  |1          |203.0             |0      |1.0       |
|1037.0|5.0  |1.0  |1          |108.0             |0      |1.0       |
|1215.0|107.0|1.0  |0          |0.0               |0      |0.0       |
|1066.0|7.0  |1.0  |1          |45.93             |0      |1.0       |
|1362.0|77.0 |1.0  |1          |87.0              |0      |1.0       |
|938.0 |21.0 |1.0  |3          |2961.82           |0      |1.0       |
|1021.0|81.0 |1.0  |1          |228.86            |0      |1.0       |
|791.0 |10.0 |1.0  |1          |-68.98            |0      |1.0       |
|917.0 |2.0  |1.0  |6          |2190.4300000000003|0      |1.0       |
|1389.0|23.0 |0.0  |2          |387.01            |0      |1.0       |
|982.0 |5.0  |1.0  |1          |19.47             |0      |1.0       |
|1232.0|13.0 |1.0  |1          |35.62             |0      |1.0       |
|963.0 |1.0  |1.0  |1          |52.44             |0      |1.0       |
|1257.0|27.0 |1.0  |1          |55.7              |0      |1.0       |
|1364.0|45.0 |1.0  |1          |118.84            |0      |1.0       |
|1371.0|229.0|1.0  |0          |0.0               |0      |1.0       |
|883.0 |10.0 |1.0  |1          |123.3             |0      |1.0       |
|748.0 |223.0|1.0  |0          |0.0               |0      |0.0       |
|1272.0|1.0  |0.0  |2          |481.42            |0      |1.0       |
|1198.0|5.0  |1.0  |1          |8.91              |0      |1.0       |
+------+-----+-----+-----------+------------------+-------+----------+
only showing top 20 rows

AUROC: 0.4415657248316914
AUPRCs: 0.9168466116389519
tp: 1912
fp: 327
fn: 351
recall: 0.8448961555457357
precision: 0.8539526574363555
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|  340|
|  1.0| 2263|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  364|
|       1.0| 2239|
+----------+-----+

cluster : 0:Train and Evaluate End
...
```