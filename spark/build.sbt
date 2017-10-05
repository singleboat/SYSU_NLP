name := "PaperRank"

version := "2.1"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "1.5.1",
    "org.apache.spark" %% "spark-mllib" % "1.5.1"
    )