package l1distopt

import breeze.linalg.SparseVector
import l1distopt.utils._
import l1distopt.solvers._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}


object driver {

  def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
      }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits", "1").toInt
    val testFile = options.getOrElse("testFile", "")
    
    // algorithm-specific inputs
    val eta = options.getOrElse("eta", "1.0").toDouble // elastic net parameter: 1.0 = lasso, 0.0 = ridge regression
    val lambda = options.getOrElse("lambda", "0.01").toDouble // regularization parameter
    val numRounds = options.getOrElse("numRounds", "200").toInt // number of outer iterations, called T in the paper
    val localIterFrac = options.getOrElse("localIterFrac", "1.0").toDouble; // fraction of local points to be processed per round, H = localIterFrac * n
    val debugIter = options.getOrElse("debugIter", "10").toInt // set to -1 to turn off debugging output
    val seed = options.getOrElse("seed", "0").toInt // set seed for debug purposes

    // print out inputs
    println("master:       " + master);          println("trainFile:    " + trainFile);
    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits);      
    println("testfile:     " + testFile);        println("eta           " + eta);       
    println("lambda:       " + lambda);          println("numRounds:    " + numRounds);       
    println("localIterFrac:" + localIterFrac);   println("debugIter     " + debugIter);       
    println("seed          " + seed);            

    // start spark context
    val conf = new SparkConf().setMaster(master)
    .setAppName("Sparse-CoCoA")
    .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)


    /**
     * Prepare Data
     */

    // read in data column-wise
    val dataAndLabels = OptUtils.loadLIBSVMDataColumn(sc, trainFile, numSplits, numFeatures)
    val data = dataAndLabels._1.cache()
    val force_cache = data.count().toInt
    val labels = dataAndLabels._2
    val n = labels.size

    val testData = {
      if (testFile != ""){ OptUtils.loadLIBSVMData(sc, testFile, numSplits, numFeatures).cache() }
      else { null }
    }

    // compute H, # of local iterations
    var localIters = (localIterFrac * numFeatures / data.partitions.size).toInt
    localIters = Math.max(localIters,1)
    val alphaInit = SparseVector.zeros[Double](numFeatures)

    // set parameters
    val params = Params(alphaInit, n, numRounds, localIters, lambda, eta)
    val debug = DebugParams(testData, debugIter, seed)


    /**
     * Run ProxCoCoA+
     */

    val finalAlphaCoCoA = ProxCoCoAp.runProxCoCoAp(data, labels, params, debug)
    sc.stop()
  }
}
