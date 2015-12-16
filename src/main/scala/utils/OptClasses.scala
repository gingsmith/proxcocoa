package l1distopt.utils

import breeze.linalg.{SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

case class LabeledPoint(val label: Double, val features: SparseVector[Double])

/** Algorithm Params
 * @param alphaInit initial weight vector (zero)
 * @param n number of data points
 * @param numRounds number of outer iterations T in the paper
 * @param localIters number of inner localSDCA iterations, H in the paper
 * @param lambda the regularization parameter
 * @param eta elastic net parameter: eta=1.0 gives lasso, eta=0.0 gives ridge regression
 */
case class Params(
    alphaInit: Vector[Double],
    n: Int,
    numRounds: Int,
    localIters: Int,
    lambda: Double,
    eta: Double)

/** Debug Params
 * @param testData
 * @param debugIter
 * @param seed
 */
case class DebugParams(
    testData: RDD[LabeledPoint],
    debugIter: Int,
    seed: Int)
