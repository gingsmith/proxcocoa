package l1distopt.utils

import breeze.linalg.{DenseVector, NumericOps, SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import scala.math._

object OptUtils {

  /**
  * Loads data stored in LIBSVM format columnwise (i.e., by feature)
  * Used for storing training dataset
  *
  * @param sc SparkContext
  * @param filename location of data
  * @param numSplits number of data splits
  * @param numFeats number of features in the dataset
  * @return
  */
  def loadLIBSVMDataColumn(sc: SparkContext, filename: String, numSplits: Int, 
    numFeats: Int): (RDD[(Int, SparseVector[Double])], DenseVector[Double]) = {

    // read in text file
    val data = sc.textFile(filename,numSplits).coalesce(numSplits)  // note: coalesce can result in data being sent over the network. avoid this for large datasets
    val numEx = data.count().toInt

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
      }.collect().sortBy(_._1)
      val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    val parsedData = data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx) {

          // parse label
          val parts = line.trim().split(' ')
          var label = parts(0).toDouble

          // parse features
          val featureArray = parts.slice(1,parts.length)
          .map(_.split(':') 
            match { case Array(i,j) => (i.toInt-1, (index, j.toDouble))}).toArray
          Iterator((label, featureArray))
        }
        else {
          Iterator()
        }
      }
    }

    // collect all of the labels
    val y = new DenseVector[Double](parsedData.map(x => x._1).collect())

    // arrange RDD by feature
    val feats = parsedData.flatMap(x => x._2.iterator)
    .groupByKey().map(x => (x._1, x._2.toArray)).map(x => (x._1, new SparseVector[Double](x._2.map(y => y._1), x._2.map(y => y._2), numEx)))
    
    // return data and labels
    println("successfully loaded training data")
    return (feats,y)
  }


  /**
  * Loads data stored in LIBSVM format
  * Used for storing test dataset
  *
  * @param sc SparkContext
  * @param filename location of data
  * @param numSplits number of data splits
  * @param numFeats number of features in the dataset
  * @return
  */
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, 
    numFeats: Int): RDD[LabeledPoint] = {

    // read in text file
    val data = sc.textFile(filename,numSplits).coalesce(numSplits)
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
      }.collect().sortBy(_._1)
      val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx){

          // parse label
          val parts = line.trim().split(' ')
          var label = parts(0).toDouble

          // parse features
          val featureArray = parts.slice(1,parts.length)
          .map(_.split(':') 
            match { case Array(i,j) => (i.toInt-1,j.toDouble)}).toArray
          var features = new SparseVector[Double](featureArray.map(x=>x._1), 
            featureArray.map(x=>x._2), numFeats)

          // create classification point
          Iterator(LabeledPoint(label, features))
        }
        else{
          Iterator()
        }
      }
    }
  }

  /**
  * Computes the primal objective function value for elastic net regression:
  *   1/(2n)||y-x'w||_2^2 + \lambda * (eta*||w||_1 + (1-eta)*.5*||w||_2^2)
  * Caution: use for debugging purposes. This is an expensive operation, 
  *   taking one full pass through the data
  *
  * @param z residual vector
  * @param w primal vector
  * @param lambda regularization parameter
  * @param eta elastic net parameter
  * @return
  */
  def computeElasticNetObjective(z: Vector[Double], w: Vector[Double], 
    lambda: Double,eta: Double): Double = {
    val err = z.norm(2) 
    val regularization = lambda * (eta * w.norm(1) + (1 - eta) * .5 * w.norm(2))
    return err * err / (2 * z.size) + regularization
  }

  /**
  * Computes the RMSE on a test dataset
  *
  * @param testData RDD of labeledpoints
  * @param w primal vector
  * @return
  */
  def computeRMSE(testData: RDD[LabeledPoint], w: Vector[Double]): Double = {
    val squared_err = testData.map(pt => pow(((pt.features dot w) - pt.label),2)).mean()
    return sqrt(squared_err)
  }

}