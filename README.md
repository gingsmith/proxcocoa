# ProxCoCoA+
A primal-dual framework for distributed L1-regularized optimization, running on [Apache Spark](spark.apache.org).

This code trains a standard least squares sparse regression with L1 or elastic net regularizer. The proxCoCoA+ framework runs on the primal optimization problem (called D in [the paper](http://arxiv.org/pdf/1512.04011)). To solve the data-local subproblems on each machine, an arbitrary solver can be used. In this example we use randomized coordinate descent as the local solver, as the L1-regularized single coordinate problems have simple closed-form solutions.

The code can be easily adapted to include other internal solvers or to solve other data-fit objectives or regularizers.

## Getting Started
How to run the code locally:

```
sbt/sbt assembly
./run-demo-local.sh
```

(For the `sbt` script to run, make sure you have downloaded CoCoA into a directory whose path contains no spaces.)

## References
The algorithmic framework is described in more detail in the following paper:

_Smith, V., Forte, S., Jordan, M.I., Jaggi, M. [L1-Regularized Distributed Optimization: A Communication-Efficient Primal-Dual Framework](http://arxiv.org/abs/1512.04011)_
