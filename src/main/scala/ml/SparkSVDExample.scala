package ml

import scala.collection.mutable
import scala.io.Source

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.graphx.Edge

/**
  * 测试样例，数据集使用ml-100k
  */
object SparkSVDExample {

  private class Encoder(maxId: Int) {
    private final val numBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(maxId), 31)
    private final val usrBit = 1 << numBits
    private final val itmBit = 2 << numBits
    private final val mask = (1 << numBits) - 1

    def encode(id: Int, usr:Boolean): Int = {
      if(usr) usrBit | id else itmBit | id
    }

    def id(encode:Int):Int = {
      encode & mask
    }
  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("Spark SVD Example").setMaster("local")
    val sc = new SparkContext(conf)

    val trainDataPath = "data/ml-100k/u.data"

    var maxId = -1
    val ratings = mutable.ArrayBuilder.make[(Int, Int, Double)]
    for(line <- Source.fromFile(trainDataPath).getLines) {
      val Array(u, v, r, _) = line.split("\t")
      val usr = u.toInt
      val itm = v.toInt
      ratings += ((usr, itm, r.toDouble))
      maxId = math.max(usr, math.max(maxId, itm))
    }

    val encoder = new Encoder(maxId)

    val ratingsEdges = ratings.result().map{case (u, v, r) =>
      Edge(encoder.encode(u, true), encoder.encode(v, false), r)
    }

    SparkSVD.run(sc.parallelize(ratingsEdges),
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
