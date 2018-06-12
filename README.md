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
please use this docker account
username:msba6212
password:hadoop123
Get Docker CE for Windows (stable)
Double-click Docker for Windows Installer to run the installer.
When the installation finishes, Docker starts automatically. The whale  in the notification area indicates that Docker is running, and accessible from a terminal.
### Install the Docker Toolbox
https://docs.docker.com/toolbox/toolbox_install_windows/
After installation , click Kitematic (Alpha) shortcut
Then click DOCKER-CLI on the left corner, you will enter a docker cli window

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
## execute the run script
```sh
root@driver:/home/msba6212# ./run_als.sh
```
## the expected log
```sh
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
18/06/12 05:54:12 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
AppParams are AppParams(25,100,20130530,20140615,20140630,10,100,0.001,1.0E-7,true,false,true,0.2,/home/msba6212/data/pcard.csv,ClusterParams(5,30,3))
positive samples count: 67334
ulimit count: 5178
mlimit count: 435
randomNegativeSamples
combinedDF count: 80368
+------+-----+-----+
|   uid|  mid|label|
+------+-----+-----+
|2690.0|196.0|  0.0|
|4691.0|250.0|  0.0|
|3118.0|102.0|  0.0|
|1079.0| 50.0|  0.0|
| 609.0|390.0|  0.0|
+------+-----+-----+
only showing top 5 rows

Start Kmeans trainning , training records count: 80368 numClusters is 5 numIterations is 30 runTimes is 3
18/06/12 05:54:49 WARN KMeans: The input data is not directly cached, which may hurt performance if its parent RDDs are also uncached.
18/06/12 05:54:55 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
18/06/12 05:54:55 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
18/06/12 05:55:10 WARN KMeans: The input data was not directly cached, which may hurt performance if its parent RDDs are also uncached.
Cluster Number:5
Cluster Centers Information Overview:
Center Point of Cluster 0:
[1207.7139822632892,71.91784822898411]
Center Point of Cluster 1:
[4380.669539881024,119.69952652664804]
Center Point of Cluster 2:
[2146.782338871291,73.91550843622728]
Center Point of Cluster 3:
[351.71188870805236,72.78753003177555]
Center Point of Cluster 4:
[3198.182203742204,86.96939708939709]
positive samples count: 8924
ulimit count: 2824
mlimit count: 260
randomNegativeSamples
combinedDF count: 10692
+------+-----+-----+
|   uid|  mid|label|
+------+-----+-----+
|1417.0| 76.0|  0.0|
| 544.0|176.0|  0.0|
|2308.0|197.0|  0.0|
|1921.0|222.0|  0.0|
|1041.0|211.0|  0.0|
+------+-----+-----+
only showing top 5 rows

Start ALS pipeline for cluster: 0
Count of cluster: 0 is 797
Split data into Training and Validation for cluster : 0:
cluster : 0: training records count: 16685
cluster : 0: validation records count: 2640
18/06/12 05:55:29 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
18/06/12 05:55:29 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
positiveDF count: 2640
validationDF count: 2640
+------+-----+-----+-------+----------+
|uid   |mid  |label|cluster|prediction|
+------+-----+-----+-------+----------+
|1623.0|173.0|1.0  |0      |0.0       |
|1472.0|7.0  |1.0  |0      |1.0       |
|1514.0|51.0 |1.0  |0      |0.0       |
|996.0 |131.0|1.0  |0      |0.0       |
|1097.0|10.0 |1.0  |0      |1.0       |
|1045.0|21.0 |1.0  |0      |1.0       |
|1603.0|2.0  |1.0  |0      |1.0       |
|904.0 |6.0  |0.0  |0      |1.0       |
|1304.0|124.0|1.0  |0      |1.0       |
|1311.0|1.0  |1.0  |0      |1.0       |
|1553.0|18.0 |1.0  |0      |1.0       |
|941.0 |3.0  |1.0  |0      |1.0       |
|998.0 |11.0 |1.0  |0      |1.0       |
|827.0 |2.0  |1.0  |0      |1.0       |
|941.0 |40.0 |1.0  |0      |1.0       |
|1127.0|131.0|1.0  |0      |1.0       |
|1183.0|21.0 |1.0  |0      |1.0       |
|998.0 |73.0 |1.0  |0      |1.0       |
|1611.0|5.0  |1.0  |0      |1.0       |
|783.0 |49.0 |1.0  |0      |1.0       |
+------+-----+-----+-------+----------+
only showing top 20 rows

AUROC: 0.4281760320376608
AUPRCs: 0.9118898551113896
tp: 1870
fp: 335
fn: 421
recall: 0.8162374508948058
precision: 0.8480725623582767
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|  349|
|  1.0| 2291|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  435|
|       1.0| 2205|
+----------+-----+

cluster : 0:Train and Evaluate End
Start ALS pipeline for cluster: 1
Count of cluster: 1 is 310
Split data into Training and Validation for cluster : 1:
cluster : 1: training records count: 1802
cluster : 1: validation records count: 464
positiveDF count: 464
validationDF count: 464
+------+-----+-----+-------+----------+
|uid   |mid  |label|cluster|prediction|
+------+-----+-----+-------+----------+
|4496.0|34.0 |1.0  |1      |0.0       |
|4177.0|42.0 |1.0  |1      |0.0       |
|3960.0|7.0  |1.0  |1      |0.0       |
|4013.0|10.0 |1.0  |1      |1.0       |
|4817.0|6.0  |1.0  |1      |0.0       |
|4869.0|95.0 |1.0  |1      |0.0       |
|3910.0|120.0|1.0  |1      |0.0       |
|4791.0|14.0 |1.0  |1      |0.0       |
|4861.0|77.0 |1.0  |1      |0.0       |
|4057.0|28.0 |1.0  |1      |1.0       |
|3880.0|4.0  |1.0  |1      |1.0       |
|3853.0|310.0|1.0  |1      |0.0       |
|4505.0|34.0 |1.0  |1      |1.0       |
|4098.0|23.0 |1.0  |1      |1.0       |
|4173.0|19.0 |1.0  |1      |1.0       |
|4265.0|54.0 |1.0  |1      |1.0       |
|4706.0|38.0 |1.0  |1      |0.0       |
|3807.0|65.0 |1.0  |1      |0.0       |
|4689.0|38.0 |1.0  |1      |0.0       |
|4335.0|164.0|1.0  |1      |0.0       |
+------+-----+-----+-------+----------+
only showing top 20 rows

AUROC: 0.46624816907302785
AUPRCs: 0.888252548525537
tp: 158
fp: 27
fn: 247
recall: 0.39012345679012345
precision: 0.8540540540540541
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|   59|
|  1.0|  405|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  279|
|       1.0|  185|
+----------+-----+

cluster : 1:Train and Evaluate End
Start ALS pipeline for cluster: 2
Count of cluster: 2 is 807
Split data into Training and Validation for cluster : 2:
cluster : 2: training records count: 12578
cluster : 2: validation records count: 1999
positiveDF count: 1999
validationDF count: 1999
+------+-----+-----+-------+----------+
|uid   |mid  |label|cluster|prediction|
+------+-----+-----+-------+----------+
|2103.0|221.0|1.0  |2      |0.0       |
|2502.0|7.0  |1.0  |2      |1.0       |
|1886.0|75.0 |0.0  |2      |1.0       |
|1726.0|51.0 |1.0  |2      |1.0       |
|1693.0|26.0 |1.0  |2      |1.0       |
|1927.0|20.0 |1.0  |2      |1.0       |
|2022.0|81.0 |1.0  |2      |1.0       |
|2357.0|124.0|1.0  |2      |0.0       |
|1739.0|146.0|1.0  |2      |0.0       |
|2100.0|58.0 |1.0  |2      |1.0       |
|1893.0|89.0 |1.0  |2      |0.0       |
|2271.0|26.0 |1.0  |2      |1.0       |
|2051.0|24.0 |1.0  |2      |1.0       |
|2070.0|31.0 |1.0  |2      |1.0       |
|2648.0|24.0 |1.0  |2      |1.0       |
|2368.0|26.0 |1.0  |2      |1.0       |
|2184.0|45.0 |1.0  |2      |1.0       |
|2610.0|12.0 |1.0  |2      |1.0       |
|2176.0|191.0|1.0  |2      |0.0       |
|2084.0|43.0 |1.0  |2      |1.0       |
+------+-----+-----+-------+----------+
only showing top 20 rows

AUROC: 0.3858960545892976
AUPRCs: 0.9140317464787537
tp: 1260
fp: 205
fn: 520
recall: 0.7078651685393258
precision: 0.8600682593856656
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|  219|
|  1.0| 1780|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  534|
|       1.0| 1465|
+----------+-----+

cluster : 2:Train and Evaluate End
Start ALS pipeline for cluster: 3
Count of cluster: 3 is 741
Split data into Training and Validation for cluster : 3:
cluster : 3: training records count: 24611
cluster : 3: validation records count: 4605
positiveDF count: 4605
validationDF count: 4605
+-----+-----+-----+-------+----------+
|uid  |mid  |label|cluster|prediction|
+-----+-----+-----+-------+----------+
|297.0|28.0 |1.0  |3      |1.0       |
|583.0|41.0 |1.0  |3      |1.0       |
|60.0 |11.0 |1.0  |3      |1.0       |
|274.0|14.0 |1.0  |3      |1.0       |
|681.0|98.0 |1.0  |3      |1.0       |
|720.0|6.0  |1.0  |3      |1.0       |
|468.0|4.0  |0.0  |3      |1.0       |
|103.0|34.0 |1.0  |3      |1.0       |
|595.0|68.0 |1.0  |3      |1.0       |
|539.0|3.0  |0.0  |3      |1.0       |
|8.0  |140.0|1.0  |3      |1.0       |
|529.0|29.0 |0.0  |3      |1.0       |
|296.0|5.0  |1.0  |3      |1.0       |
|114.0|223.0|1.0  |3      |1.0       |
|179.0|2.0  |1.0  |3      |1.0       |
|372.0|51.0 |1.0  |3      |1.0       |
|750.0|14.0 |1.0  |3      |1.0       |
|51.0 |134.0|1.0  |3      |0.0       |
|593.0|261.0|1.0  |3      |1.0       |
|282.0|125.0|1.0  |3      |1.0       |
+-----+-----+-----+-------+----------+
only showing top 20 rows

AUROC: 0.4539551985037831
AUPRCs: 0.9103154842032962
tp: 3451
fp: 665
fn: 470
recall: 0.8801326192297884
precision: 0.8384353741496599
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|  684|
|  1.0| 3921|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  489|
|       1.0| 4116|
+----------+-----+

cluster : 3:Train and Evaluate End
Start ALS pipeline for cluster: 4
Count of cluster: 4 is 511
Split data into Training and Validation for cluster : 4:
cluster : 4: training records count: 5480
cluster : 4: validation records count: 984
positiveDF count: 984
validationDF count: 984
+------+-----+-----+-------+----------+
|uid   |mid  |label|cluster|prediction|
+------+-----+-----+-------+----------+
|2795.0|163.0|1.0  |4      |0.0       |
|3295.0|12.0 |0.0  |4      |1.0       |
|3100.0|9.0  |1.0  |4      |1.0       |
|3212.0|13.0 |1.0  |4      |0.0       |
|2741.0|34.0 |1.0  |4      |1.0       |
|3747.0|43.0 |1.0  |4      |1.0       |
|2687.0|169.0|1.0  |4      |0.0       |
|3323.0|6.0  |1.0  |4      |1.0       |
|3283.0|60.0 |1.0  |4      |1.0       |
|2867.0|28.0 |0.0  |4      |1.0       |
|3547.0|51.0 |1.0  |4      |1.0       |
|3338.0|7.0  |1.0  |4      |1.0       |
|2902.0|2.0  |1.0  |4      |1.0       |
|3774.0|4.0  |1.0  |4      |1.0       |
|2809.0|12.0 |1.0  |4      |1.0       |
|2884.0|4.0  |1.0  |4      |1.0       |
|2884.0|32.0 |1.0  |4      |1.0       |
|3398.0|6.0  |1.0  |4      |1.0       |
|3129.0|25.0 |1.0  |4      |1.0       |
|2686.0|23.0 |1.0  |4      |1.0       |
+------+-----+-----+-------+----------+
only showing top 20 rows

AUROC: 0.47142563470038024
AUPRCs: 0.9070367163546751
tp: 626
fp: 110
fn: 220
recall: 0.7399527186761229
precision: 0.8505434782608695
label distribution:
+-----+-----+
|label|count|
+-----+-----+
|  0.0|  138|
|  1.0|  846|
+-----+-----+

prediction distribution:
+----------+-----+
|prediction|count|
+----------+-----+
|       0.0|  248|
|       1.0|  736|
+----------+-----+

cluster : 4:Train and Evaluate End
total time: 278.6911308
```

**Free Software, Hell Yeah!**