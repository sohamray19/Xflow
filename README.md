# Xflow

A Real-Time Tweet Streaming Pipeline with Deep Learning Sentiment Analysis Model for instant scoring.
- The Real-Time Tweet Streaming Pipeline is built using Apache Flume, Apache Kafka & Spark-Streaming.
- The LSTM based Sentiment Analysis Model is built using Keras with Tensorflow Backend.(Uses Word-Embeddings)
- This Model is exposed as a RESTful Service which enables flexible usage.

## Usage
Clone this repo on your system. Ensure maven is installed on your system for building it.
Go to root directory of the project and run.
```sh 
mvn clean install
```
* This trains the LSTM based Deep Learning Sentiment Analysis Model and exports it as a RESTful service.</br>
* The training dataset is 'Sentiment Analysis Dataset.csv' downloaded from https://bit.ly/1TVSjsF .</br>
* The RESTful Service is hosted on http://localhost:5003/ </br>

The Sentiment prediction for any Tweet/Sentence can be obtained by sending a POST request given as follows:
```sh
curl --header "Content-Type: application/json" --request POST --data '{"data":"YOUR TWEET HERE"}' http://localhost:5003/
```
### For Real-Time sentiment analysis of tweets, the streaming data pipeline is built as follows:

1. Download Confluent Open Source from https://www.confluent.io/download/ (Tested on v5.0).
Extract it and inside the directory, run the following command: 
```sh
bin/confluent start
```
- This will start Kafka, Schema Registry, Zookeeper etc.

2. Download and extract flume binary file from https://flume.apache.org/download.html </br>

3. Clone cloudera twitter-example-github repo from https://github.com/cloudera/cdh-twitter-example

- The flume-sources directory contains a Maven project with Cloudera custom Flume source designed to connect to the Twitter Streaming API and ingest tweets in a raw JSON format.

4. To build the flume-sources JAR, from the root of the git repository:

```
$ cd flume-sources  
$ mvn package
$ cd ..
```
5. Add the JAR to the Flume classpath. Copy 
```flume-sources-1.0-SNAPSHOT.jar``` to ```apache-flume-latest-version-bin/plugins.d/twitter-streaming/lib/```

6. Tweets are ingested in raw JSON format and pushed to a Kafka sink. Flume configurations are set in FlumeConfig.conf and Agent is set as Twitter Agent.

- To start flume, run 
```
bin/flume-ng agent --conf conf --conf-file FlumeConfig.conf --name TwitterAgent -Dflume.root.logger=INFO,console
```
7. Run Score.scala

- Spark Streaming is used to consume tweets from the Kafka queue. Sentiment prediction for each of the tweets is obtained by sending a POST request to the RESTful Service as described above.
