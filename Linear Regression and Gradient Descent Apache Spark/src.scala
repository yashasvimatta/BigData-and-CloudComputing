// Databricks notebook source
// Yashasvi Matta 
// 1002091131
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector, Matrices, Matrix}
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseMatrix, inv, DenseVector}
import org.apache.spark.{SparkConf, SparkContext}
// creating example matrix and vector 
val dataX: Array[Vector] = Array(
  Vectors.dense(1.0, 2.0),
  Vectors.dense(3.0, 4.0),
  Vectors.dense(5.0, 6.0)
)
val dataY: Array[Double] = Array(7.0, 8.0, 9.0)
val X: RDD[Vector] = sc.parallelize(dataX)
val Y: RDD[Double] = sc.parallelize(dataY)
// calculating X^T X using the outer product method
val x: Matrix = X.map { x =>
  val xArray = x.toArray
  val outerProduct = Array.ofDim[Double](xArray.length, xArray.length)
  for (i <- xArray.indices; j <- xArray.indices) {
    outerProduct(i)(j) = xArray(i) * xArray(j)
  }
  Matrices.dense(xArray.length, xArray.length, outerProduct.flatten)
}.reduce((m1, m2) => Matrices.dense(m1.numRows, m1.numCols, m1.toArray.zip(m2.toArray).map { case (a, b) => a + b }))
// compute the result matrix to a Breeze Dense Matrix and compute the inverse
val bX: DenseMatrix[Double] = new DenseMatrix(x.numRows, x.numCols, x.toArray)
val iX: DenseMatrix[Double] = inv(bX)
// compute X transpose y
val xTy: DenseVector[Double] = DenseVector(X.zip(Y).map { case (x, y) =>
  val xArray = x.toArray
  xArray.map(_ * y)
}.reduce((v1, v2) => v1.zip(v2).map { case (a, b) => a + b }))
// get the coefficients
val theta: DenseVector[Double] = iX * xTy
// print theta values
println(s"X: ${X.collect().map(_.toArray.mkString("[", ", ", "]")).mkString(", ")}")
println(s"Y: ${Y.collect().mkString("[", ", ", "]")}")
println(s"X^T X: ${x.toString()}")
println(s"X^T y: ${xTy.data.mkString("[", ", ", "]")}")
println(s"theta: ${theta.data.mkString(", ")}")


// COMMAND ----------

// Bonus Question
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{DenseVector}
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD
def calculateGradientComponent(weights: Vector, features: Vector, actualValue: Double): Vector = {
  val fVector = DenseVector(features.toArray)
  val wVector = DenseVector(weights.toArray)
  val prediction = wVector.dot(fVector)
  val gradient = (prediction - actualValue) * fVector
  Vectors.dense(gradient.toArray)
}
val weightsExample1 = Vectors.dense(0.4, 0.3, 0.1)
val dataPoint1 = LabeledPoint(1.5, Vectors.dense(1.0, 2.0, 3.0))
println("Gradient Component for Data 1: " + calculateGradientComponent(weightsExample1, dataPoint1.features, dataPoint1.label))
val weightsExample2 = Vectors.dense(0.2, 0.4, 0.1)
val data2 = LabeledPoint(2.5, Vectors.dense(4.0, 5.0, 6.0))
println("Gradient Component for Data 2: " + calculateGradientComponent(weightsExample2, data2.features, data2.label))
def predictLabel(weights: Vector, labeledData: LabeledPoint): (Double, Double) = {
  val prediction = weights dot labeledData.features
  (labeledData.label, prediction)
}
val predicted1 = predictLabel(weightsExample1, dataPoint1)
val predicted2 = predictLabel(weightsExample2, data2)
val predictionsRDD = sc.parallelize(Seq(predicted1, predicted2))
println("Predictions: " + "\n" + predicted1 + "\n" + predicted2)
def calculateRMSE(predictions: RDD[(Double, Double)]): Double = {
  val squaredErrors = predictions.map{ case(actual, predicted) => 
    val error = actual - predicted
    error * error
  }.mean()

  math.sqrt(squaredErrors)
}
val rmse = calculateRMSE(predictionsRDD)
println(s"RMSE: $rmse")
def performGradientDescent(trainingData: RDD[LabeledPoint], iterationCount: Int): (DenseVector[Double], Array[Double]) = {
  val numFeatures = trainingData.first().features.size
  var weights = DenseVector.zeros[Double](numFeatures)
  var alpha = 0.1
  val total = trainingData.count()
  val iterationErrors = ArrayBuffer[Double]()
  for (iter <- 1 to iterationCount) {
    val gradientSum = trainingData.map(point => {
      val featureVec = new DenseVector(point.features.toArray)
      val predictionError = weights.dot(featureVec) - point.label
      predictionError * featureVec
    }).reduce(_ + _)
    weights -= (alpha / math.sqrt(iter)) * gradientSum
    val currentRMSE = calculateRMSE(
      trainingData.map(point => {
        val features = new DenseVector(point.features.toArray)
        (point.label, weights.dot(features))
      })
    )
    iterationErrors += currentRMSE

    alpha = alpha / (total * math.sqrt(iter))
  }
  (weights, iterationErrors.toArray)
}
val trainingDataset = sc.parallelize(Seq(
  LabeledPoint(1.2, Vectors.dense(1.0, 2.0)),
  LabeledPoint(2.3, Vectors.dense(3.0, 1.0)),
  LabeledPoint(3.4, Vectors.dense(2.0, 3.0)), 
  LabeledPoint(2.2, Vectors.dense(2.0, 2.0))
))
val (finalWeights, iterationErrorValues) = performGradientDescent(trainingDataset, 5)
println("Final Weights: " + finalWeights)
println("Error per Iteration: " + iterationErrorValues.mkString(","))


// COMMAND ----------


