package com.intel.analytics.zoo.benchmark.inference


case class PerfParams (
                        batchSize: Int = 4,
                        iteration: Int = 10,
                        model: String = "resnet50",
                        quantize: Boolean = false,
                        outputFormat: String = "nc",
                        numInstance: Int = 1,
                        coreNumber: Int = 18
                      )
