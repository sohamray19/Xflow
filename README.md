# Xflow
Real time Twitter streaming pipeline with Deep Learning Sentiment Analysis model for instant scoring
Download and extract flume binary file from https://flume.apache.org/download.html
To create custom flume source, clone cloudera twitter example github repo from https://github.com/cloudera/cdh-twitter-example

The flume-sources directory contains a Maven project with a custom Flume source designed to connect to the Twitter Streaming API and ingest tweets in a raw JSON format into HDFS.

To build the flume-sources JAR, from the root of the git repository:

```
$ cd flume-sources  
$ mvn package
$ cd ..
```
Add the JAR to the Flume classpath
Copy flume-sources-1.0-SNAPSHOT.jar to ```apache-flume-latest-version-bin/plugins.d/twitter-streaming/lib/flume-sources-1.0-SNAPSHOT.jar```

