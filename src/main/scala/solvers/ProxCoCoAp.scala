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
   * We set sigma'=K (number of partitions), and gamma=1
   *
   * @param data RDD of all data columns (columns of the data matrix A in the paper)
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
    var alpha = params.alphaInit // primal weight vector
    var w = labels.copy // residual vector w = A * alpha - b
    var elapsedTime = 0.0

    // run for numRounds rounds
    for(t <- 1 to params.numRounds){

      // start time
      val tstart = System.currentTimeMillis

      // find updates to alpha, w
      val updates = dataArr.mapPartitions(
        localCD(_, alpha, w, params.localIters, params.eta, params.lambda, params.n,
          parts, debug.seed+t), preservesPartitioning=true).persist()
      val primalUpdates = updates.map(kv => kv._1).treeReduce(_ + _)
      alpha += primalUpdates
      val residualUpdates = updates.map(kv => kv._2).treeReduce(_ + _)
      w += residualUpdates

      // optionally calculate optimization objective and test RMSE
      elapsedTime = elapsedTime + (System.currentTimeMillis-tstart)
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println(t + "," + elapsedTime + "," 
          + OptUtils.computeElasticNetObjective(alpha, w, params.lambda, params.eta))
        if(debug.testData != null) println("Test RMSE: " 
          + OptUtils.computeRMSE(debug.testData, alpha))
      }
    }

    // return final weight vector
    alpha
  }

  /**
   * This is an implementation of a local solver, here coordinate descent (CD),
   * that takes information from other workers into account through the shared vector w.
   * Here we perform coordinate updates for the elastic net primal objective.
   *
   * @param localDataItr the local data, split by feature
   * @param alphaInit current variables x (called alpha in the paper)
   * @param wInit current residual vector w = A * x - b
   * @param localIters number of local coordinates to update
   * @param eta elastic net parameter
   * @param lambda regularization parameter
   * @param n total number of columns of the data matrix A (not only local)
   * @param sigma' subproblem parameter
   * @param seed
   */
  def localCD(
      localDataItr: Iterator[Array[(Int, SparseVector[Double])]],
      alphaInit: Vector[Double],
      wInit: Vector[Double],
      localIters: Int,
      eta: Double,
      lambda: Double,
      n: Int,
      sigma: Int,
      seed: Int): Iterator[(Vector[Double], Vector[Double])] = {

    val localData = localDataItr.next()
    val alpha = alphaInit.copy
    var w = wInit.copy
    val nLocal = localData.length
    val r = new scala.util.Random(seed)

    // perform local udpates
    var i = 0
    val denom = lambda * n * (1-eta)
    while(i < localIters) {

      // gather current feature
      val idx = r.nextInt(nLocal)
      val currFeat = localData(idx)._2
      val j = localData(idx)._1
      val alphaj_old = alpha(j)

      // calculate update
      val aj = pow(currFeat.norm(2), 2)
      val grad = ((currFeat dot w) / (aj * sigma + denom)) + (alphaj_old / (denom / (aj * sigma) + 1.0))

      // apply soft thresholding
      val threshold = (n * lambda * eta) / (aj * sigma + denom)
      alpha(j) = (signum(grad) * max(0.0, abs(grad) - threshold))

      // find change in primal vector
      val diff = alphaj_old - alpha(j)

      // update residual vector
      w += currFeat * (diff * (sigma))

      i += 1
    }

    // return changes to alpha, w
    val deltaAlpha = alpha - alphaInit
    val deltaW = (w - wInit) / (sigma.toDouble)
    Iterator((deltaAlpha, deltaW))
  }

}
