ThisBuild / scalaVersion := "2.11.12"
ThisBuild / organization := "org.rise-lang"

lazy val shine = (project in file("."))
  .dependsOn(executor, rise, arithExpr)
  .settings(
    name    := "shine",
    version := "1.0",

    javaOptions ++= Seq("-Djava.library.path=lib/executor/lib/Executor/build", "-Xss8m"),

    scalacOptions ++= Seq(
      "-Xfatal-warnings",
      "-Xlint",
      "-Xmax-classfile-name", "100",
      "-unchecked",
      "-deprecation",
      "-feature",
      "-language:reflectiveCalls"
    ),

    fork := true,

    // Scala libraries
    libraryDependencies += "org.scala-lang" % "scala-reflect" % scalaVersion.value,
    libraryDependencies += "org.scala-lang" % "scala-compiler" % scalaVersion.value,
    libraryDependencies += "org.scala-lang" % "scala-library" % scalaVersion.value,
    libraryDependencies += "org.scala-lang.modules" %% "scala-xml" % "1.0.5",

    // JUnit
    libraryDependencies += "junit" % "junit" % "4.11",

    // Scalatest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.5" % "test",

    // Silencer: Scala compiler plugin for warning suppression
    libraryDependencies ++= Seq(
      compilerPlugin("com.github.ghik" %% "silencer-plugin" % "1.4.0"),
      "com.github.ghik" %% "silencer-lib" % "1.4.0" % Provided
    )
  )

lazy val executor   = (project in file("lib/executor"))

lazy val rise       = (project in file("lib/rise"))

lazy val arithExpr  = (project in file("lib/rise/lib/arithexpr"))