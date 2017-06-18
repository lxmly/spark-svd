package ml

import scala.collection.mutable
import scala.io.Source

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx.Edge

/**
  * 测试样例，数据集使用ml-100k
  */
object SparkSVDExample {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Spark SVD Example").setMaster("local")
    val sc = new SparkContext(conf)

    val trainDataPath = "data/ml-100k/u.data"

    val edges = mutable.ArrayBuilder.make[Edge[Double]]
    for(line <- Source.fromFile(trainDataPath).getLines) {
      val Array(u, v, r, _) = line.split("\t")
      edges += Edge(u.toLong, v.toLong, r.toDouble)
    }

    SparkSVD.run(sc.parallelize(edges.result()),
      new SparkSVD.Conf(
        rank = 30,
        maxIters = 200,
        minVal = 1,
        maxVal = 5,
        gamma1 = 0.01,
        gamma2 = 0.01,
        lambda1 = 0.05,
        lambda2 = 0.05
      ))
  }
}
