package l1distopt.solvers

import breeze.linalg.{SparseVector, Vector}
import l1distopt.utils._
import java.security.SecureRandom
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.math._

object ProxCoCoAp {

  /**
    * ProxCoCoAp - A primal-dual framework for L1-reguarlized distributed optimization.
    * Here implemented for Elastic Net regularized least squares regression
    * (including Lasso and Ridge as special cases).
    * Uses randomized coordinate descent as the internal local method.
    *
    * @param data RDD of all data columns (columns of the matrix A in the paper)
    * @param labels Array of data labels
    * @param params algorithmic parameters
    * @param debug systems/debugging parameters
    * @return
    */
  def runProxCoCoAp(
    data: RDD[(Int, SparseVector[Double])],
    labels: Vector[Double],
    params: Params,
    debug: DebugParams) : Vector[Double] = {
    
    // prepare to run framework
    val dataArr = data.mapPartitions(x => (Iterator(x.toArray))).cache()
    val parts = dataArr.count().toInt
    println("\nRunning ProxCoCoA+ on " + params.n + " data columns, distributed over "
      + parts + " workers")
    var x = params.xInit // primal weight vector (called alpha in the paper)
    var z = labels.copy // residual vector z = A * x - b (called w in the paper)
    var elapsedTime = 0.0

    // run for numRounds rounds
    for(t <- 1 to params.numRounds){

      // start time
      val tstart = System.currentTimeMillis

      // find updates to x, z
      val updates = dataArr.mapPartitions(
        localCD(_, x, z, params.localIters, params.eta, params.lambda, params.n,
          parts, debug.seed+t), preservesPartitioning=true).persist()
      val primalUpdates = updates.map(kv => kv._1).treeReduce(_ + _)
      x += primalUpdates
      val residualUpdates = updates.map(kv => kv._2).treeReduce(_ + _)
      z += residualUpdates

      // optionally calculate primal objective and test rmse
      elapsedTime = elapsedTime + (System.currentTimeMillis-tstart)
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println(t + "," + elapsedTime + "," 
          + OptUtils.computeElasticNetObjective(x, z, params.lambda, params.eta))
        if(debug.testData != null) println("Test RMSE: " 
          + OptUtils.computeRMSE(debug.testData, x))
      }
    }

    // return final weight vector
    x
  }

  /**
    * This is an implementation of a local solver, here coordinate descent (CD),
    * that takes information from other workers into account through the shared vector z.
    * Here we perform coordinate updates for the elastic net primal objective.
    *
    * @param localDataItr the local data, split by feature
    * @param xInit current variables x (called alpha in the paper)
    * @param zInit current residual vector z = A * x - b (called w in the paper)
    * @param localIters number of local coordinates to update
    * @param eta elastic net parameter
    * @param lambda regularization parameter
    * @param n number of data examples
    * @param k number of splits
    * @param seed
    */
  def localCD(
               localDataItr: Iterator[Array[(Int, SparseVector[Double])]],
               xInit: Vector[Double],
               zInit: Vector[Double],
               localIters: Int,
               eta: Double,
               lambda: Double,
               n: Int,
               k: Int,
               seed: Int): Iterator[(Vector[Double], Vector[Double])] = {

    val localData = localDataItr.next()
    val x = xInit.copy
    var z = zInit.copy
    val nLocal = localData.length
    val r = new scala.util.Random(seed)

    // perform local udpates
    var i = 0
    val denom = 1.0 + (lambda * (1.0 - eta))
    while(i < localIters) {

      // gather current feature
      val idx = r.nextInt(nLocal)
      val currFeat = localData(idx)._2
      val j = localData(idx)._1
      val xj_old = x(j)

      // calculate update
      val aj = pow(currFeat.norm(2), 2)
      val grad = ((currFeat dot z) / (aj * k)) + xj_old

      // apply soft thresholding
      val threshold = (n * lambda / (aj * k)) * eta
      x(j) = (signum(grad) * max(0.0, abs(grad) - threshold)) / denom

      // find change in primal vector
      val diff = xj_old - x(j)

      // update residual vector
      z += currFeat * (diff * (k))

      i += 1
    }

    // return changes to x, z
    val deltaX = x - xInit
    val deltaZ = (z - zInit) / (k.toDouble)
    Iterator((deltaX, deltaZ))
  }

}
