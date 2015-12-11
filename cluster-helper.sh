#!/bin/bash

/root/spark/bin/spark-submit \
--master `cat /root/spark-ec2/cluster-url` \
--class "l1distopt.driver" \
--driver-memory 8G \
--driver-class-path target/scala-2.10/proxcocoa-assembly-0.1.jar \
--driver-java-options "-XX:+UseG1GC -Xss8m" \
target/scala-2.10/proxcocoa-assembly-0.1.jar \
"$@"
