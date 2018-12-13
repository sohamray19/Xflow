# Xflow
Real time Twitter streaming pipeline with Deep Learning Sentiment Analysis model for instant scoring
1. Download Confluent Open Source from https://www.confluent.io/download/ (Tested on v5.0).
Extract it and inside the directory, run the following command: ```bin/confluent start```
This will start Kafka, Schema Registry, Zookeeper etc.
2. Download and extract flume binary file from https://flume.apache.org/download.html
3a. To create custom flume source, clone cloudera twitter-example-github repo from https://github.com/cloudera/cdh-twitter-example

The flume-sources directory contains a Maven project with CLoudera custom Flume source designed to connect to the Twitter Streaming API and ingest tweets in a raw JSON format into HDFS.

3b. To build the flume-sources JAR, from the root of the git repository:

```
$ cd flume-sources  
$ mvn package
$ cd ..
```
3c. Add the JAR to the Flume classpath. Copy flume-sources-1.0-SNAPSHOT.jar to ```apache-flume-latest-version-bin/plugins.d/twitter-streaming/lib/flume-sources-1.0-SNAPSHOT.jar```

3d. We ingest tweets in raw JSON format and push them to a Kafka sink. Flume configurations are set in FlumeConfig.conf and Agent is set as Twitter Agent. To start flume, run 
```
bin/flume-ng agent --conf conf --conf-file FlumeConfig.conf --name TwitterAgent -Dflume.root.logger=INFO,console
```


