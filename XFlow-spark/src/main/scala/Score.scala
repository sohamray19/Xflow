import org.apache.cxf.jaxrs.client.WebClient
import org.apache.http.HttpHeaders
import com.google.gson.Gson
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import org.apache.spark.sql.streaming.StreamingQuery
import org.apache.spark.sql.{DataFrame, Dataset, SQLContext, SparkSession}
import org.apache.spark.sql.Row
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils}
import org.apache.spark.streaming.{Durations, Seconds, StreamingContext}
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.spark_project.guava.eventbus.Subscribe



class DT (var data: String) {
  override def toString = data
}

object Score {

  def hit(clause:String) ={
    val data = new DT(clause)
    val dataAsJson = new Gson().toJson(data)
    val client = WebClient.create("http://localhost:5003/")
    client.header(HttpHeaders.CONTENT_TYPE, "application/json")

//    print(s"\n\nclient obj: $client")
//    print(s"\n\njson obj: $dataAsJson")

    val r = client.post(dataAsJson)
    println(r.getClass)
    val b = r.readEntity(classOf[String])
    println(b)
  }


  def throwString(tweet:String):String={
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    val parsedJSON = mapper.readValue[Map[String, Object]](tweet)
    parsedJSON("text").toString
  }

  def main(args: Array[String]) {

    Logger.getLogger("akka").setLevel(Level.OFF)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("org").setLevel(Level.ERROR)


    val kafkaParams = Map[String, String](
      "bootstrap.servers" -> "localhost:9092,localhost:9092",
      "key.deserializer" -> classOf[StringDeserializer].getName,
      "value.deserializer" -> classOf[StringDeserializer].getName,
      "group.id" -> "customer-group1",
      "schema.registry.url" -> "http://127.0.0.1:8081",
      "auto.offset.reset" -> "latest"
    )

    val topics = List("topic_random")
    val sparkConf = new SparkConf().setAppName("KSS").setMaster("local[*]")
    val ssc = new StreamingContext(sparkConf, Seconds(2))
    ssc.checkpoint("checkpoint")

    val stream = KafkaUtils.createDirectStream[String, String](
      ssc,
      PreferConsistent,
      ConsumerStrategies.Subscribe[String, String](topics, kafkaParams)
    )

    var tweetstream = stream.map(record => record.value())
    tweetstream.foreachRDD(recordsRDD=>recordsRDD.foreach(record=>
      hit(throwString(record))))

    ssc.start()
    ssc.awaitTermination()

  }
}