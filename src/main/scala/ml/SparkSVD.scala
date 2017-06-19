package ml

import scala.util.Random
import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.spark.Logging
import org.apache.spark.graphx._
import org.apache.spark.rdd._


/**
  * SVD算法的Spark实现
  * 其中 \hat r_{ui} = \mu + b_i + b _u+q_{i}^Tp_{u}
  */
object SparkSVD extends Logging{

  /**
    * @param rank 隐式向量维度
    * @param maxIters 迭代次数
    * @param minVal 预测下限
    * @param maxVal 预测上限
    * @param gamma1 b_*学习速率
    * @param gamma2 p、q学习速率
    * @param lambda1 b_*正则系数
    * @param lambda2 p、q正则系数
    * @param verbose 是否输出RMSE
    */
  class Conf(
              var rank: Int,
              var maxIters: Int,
              var minVal: Double,
              var maxVal: Double,
              var gamma1: Double,
              var gamma2: Double,
              var lambda1: Double,
              var lambda2: Double,
              var verbose: Boolean = true)
    extends Serializable


  def run(edges: RDD[Edge[Double]], conf: Conf) {

    // 初始化隐式因子
    def initFactor(rank: Int): (Array[Double], Double) = {
      val v1 = Array.fill(rank)(Random.nextDouble())
      (v1, 0)
    }

    edges.cache()
    val (rs, rc) = edges.map(e => (e.attr, 1L)).reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val u = rs / rc

    var g = Graph.fromEdges(edges, initFactor(conf.rank)).cache()
    materialize(g)
    edges.unpersist()

    // 初始化偏见
    val t0 = g.aggregateMessages[(Long, Double)](
      ctx => {
        ctx.sendToSrc((1L, ctx.attr)); ctx.sendToDst((1L, ctx.attr))
      },
      (g1, g2) => (g1._1 + g2._1, g1._2 + g2._2))

    val gJoinT0 = g.outerJoinVertices(t0) {
      (vid: VertexId, vd: (Array[Double], Double),
       msg: Option[(Long, Double)]) =>
        (vd._1, msg.get._2 / msg.get._1 - u)
    }.cache()
    materialize(gJoinT0)
    g.unpersist()
    g = gJoinT0

    def sendMsgTrainF(conf: Conf, u: Double)
                     (ctx: EdgeContext[
                       (Array[Double], Double),
                       Double,
                       (Array[Double], Double, Int)]) {
      val (usr, itm) = (ctx.srcAttr, ctx.dstAttr)
      val (p, q) = (usr._1, itm._1)
      val rank = p.length
      var pred = u + usr._2 + itm._2 + blas.ddot(rank, q, 1, p, 1)
      val err = ctx.attr - pred

      val updateP = q.clone()
      blas.dscal(rank, err * conf.gamma2, updateP, 1)
      blas.daxpy(rank, -conf.lambda2 * conf.gamma2, p, 1, updateP, 1)

      val updateQ = p.clone()
      blas.dscal(rank, err * conf.gamma2, updateQ, 1)
      blas.daxpy(rank, -conf.lambda2 * conf.gamma2, q, 1, updateQ, 1)

      ctx.sendToSrc((updateP, (err - conf.lambda1 * usr._2) * conf.gamma1, 1))
      ctx.sendToDst((updateQ, (err - conf.lambda1 * itm._2) * conf.gamma1, 1))
    }

    for (i <- 0 until conf.maxIters) {
      g.cache()
      val t2 = g.aggregateMessages(
        sendMsgTrainF(conf, u),
        (g1: (Array[Double], Double, Int), g2: (Array[Double], Double, Int)) => {
          val out1 = g1._1.clone()
          blas.daxpy(out1.length, 1.0, g2._1, 1, out1, 1)
          (out1, g1._2 + g2._2, g1._3 + g2._3)
        })
      val gJoinT2 = g.outerJoinVertices(t2) {
        (vid: VertexId,
         vd: (Array[Double], Double),
         msg: Option[(Array[Double], Double, Int)]) => {
          val out1 = vd._1.clone()
          blas.daxpy(out1.length, 1.0 / msg.get._3, msg.get._1, 1, out1, 1)
          (out1, vd._2 + msg.get._2 / msg.get._3)
        }
      }.cache()

      materialize(gJoinT2)
      g.unpersist()
      g = gJoinT2

      if(conf.verbose) {
        calculateIterationRMSE(g)
      }
    }

    def calculateIterationRMSE(g: Graph[(Array[Double], Double), Double]): Unit = {
      val rmse = math.sqrt(g.triplets.map { ctx =>
        val (usr, itm) = (ctx.srcAttr, ctx.dstAttr)
        val (p, q) = (usr._1, itm._1)
        var pred = u + usr._2 + itm._2 + blas.ddot(q.length, q, 1, p, 1)
        pred = math.max(pred, conf.minVal)
        pred = math.min(pred, conf.maxVal)
        (ctx.attr - pred) * (ctx.attr - pred)
      }.reduce(_ + _) / g.edges.count())

      println(s"this iteration rmse:$rmse")
    }

    g.unpersist()
  }

  /**
    * 触发Action
    */
  private def materialize(g: Graph[_, _]): Unit = {
    g.vertices.count()
    g.edges.count()
  }

}
