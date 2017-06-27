import sbt._
import Keys._
import sbtassembly.AssemblyPlugin.autoImport._
import sbtassembly.MergeStrategy

import com.lightbend.paradox.sbt.{ParadoxPlugin}
import com.lightbend.paradox.sbt.ParadoxPlugin.autoImport._
import laika.sbt.LaikaSbtPlugin.{LaikaPlugin, LaikaKeys}
import LaikaKeys._

import _root_.java.nio.file.Files

object MandolinBuild extends Build {

  lazy val root = Project(id = "mandolin", base = file(".")).
                            settings(rootSettings:_*).
			    enablePlugins(ParadoxPlugin).
			    settings(
			      paradoxTheme := Some(builtinParadoxTheme("generic")),
			      paradoxGroups := Map("Platforms" -> Seq("Linux", "Mac"))).
                            aggregate(mandolinCore, mandolinSpark, mandolinMx)

  lazy val mandolinCore = Project(id = "mandolin-core", base = file("mandolin-core")).
                            settings(coreSettings:_*).
                            settings(coreDependencySettings:_*).
                            settings(assemblyProjSettings("core"):_*).
                            settings(siteSettings:_*).
                            settings(net.virtualvoid.sbt.graph.Plugin.graphSettings: _*)


  // The MXNet library comes pre-built and resides in mandolin/mandolin-mx/lib
  // In addition, the native code is pre-built in mandolin/mandolin-mx/native  
  lazy val mandolinMx = Project(id = "mandolin-mx", base = file("mandolin-mx")).
                            settings(mxNetSettings("mx"):_*).
                            //settings(mxNetDependencySettings:_*).
                            settings(assemblyProjSettings("mx"):_*).
                            //settings(siteSettings:_*).
                            settings(net.virtualvoid.sbt.graph.Plugin.graphSettings: _*) dependsOn(mandolinCore)

  lazy val mandolinSpark = Project(id = "mandolin-spark", base = file("mandolin-spark")).
                            settings(sparkSettings:_*).
                            settings(sparkDependencySettings:_*).
                            settings(assemblyProjSettings("spark"):_*).
                            //settings(siteSettings:_*).
                            settings(net.virtualvoid.sbt.graph.Plugin.graphSettings: _*) dependsOn(mandolinCore, mandolinMx)

  val mainVersion = "0.3.5"
  

  def rootSettings = sharedSettings ++ Seq(
    name := "mandolin"
  )

  def sharedSettings : Seq[Setting[_]] = Defaults.defaultSettings ++ Seq(
    organization := "org.mitre.mandolin",
    version := mainVersion+"-SNAPSHOT",
    scalaVersion := "2.11.8",
    crossScalaVersions := Seq("2.10.5","2.11.8"),
    publishTo := {
       val nexus = "https://oss.sonatype.org/"
       if (isSnapshot.value)
         Some("snapshots" at nexus + "content/repositories/snapshots")
       else
         Some("releases" at nexus + "service/standalone/staging/deploy/maven2")
    },
    publishMavenStyle := true,    
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false },
    licenses := Seq("Apache 2" -> url("http://www.apache.org/licenses/LICENSE-2.0")),
    homepage := Some(url("https://github.com/project-mandolin/mandolin.git")),
    pomExtra in Global := {
      <scm>
        <connection>scm:git:github.com:project-mandolin/mandolin.git</connection>
        <url>git@github.com/project-mandolin/mandolin.git</url>
      </scm>
      <developers>
        <developer>
          <id>wellner</id>
          <name>Ben Wellner</name>
          <url>https://github.com/project-mandolin/mandolin.git</url>
        </developer>
      </developers>
    },
    resolvers += "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
    resolvers += Resolver.url("Typesafe Release Repository",url("http://repo.typesafe.com/typesafe/releases/"))(Resolver.ivyStylePatterns),
    resolvers += "Akka Repository" at "http://repo.akka.io/releases/",
    resolvers += "Secured Central Repository" at "https://repo1.maven.org/maven2",
    resolvers += "Snapshot Repo" at "https://oss.sonatype.org/content/repositories/snapshots/",
    resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository",
    externalResolvers := Resolver.withDefaultResolvers(resolvers.value, mavenCentral = false),
    javacOptions ++= Seq("-source","1.7","-target","1.7"),
    scalacOptions in (Compile, doc) ++= Seq("-doc-root-content", baseDirectory.value+"/src/root-doc.txt", "-unchecked")
  )  

  def coreSettings : Seq[Setting[_]] = sharedSettings ++ Seq(
    name := "mandolin-core"    
  )

  def sparkSettings : Seq[Setting[_]] = sharedSettings ++ Seq(  
    name := "mandolin-spark"
  )

  def mxNetSettings(subProj: String) : Seq[Setting[_]] = sharedSettings ++ Seq(
    name := "mandolin-mx",
    unmanagedClasspath in Compile <++= baseDirectory map { base =>
      val lib = base / "lib"
      val libFiles = lib ** "*.jar"
      libFiles.get
    },
    // force the new .jar files in the lib directory to be added to classpath prior to compiling
    compile in Compile <<= (compile in Compile) dependsOn(unmanagedClasspath in Compile)
    /*
    assemblyLinuxCoreTask := {      
      Def.sequential(      
        Def.task { linuxCoreTask }
      ).value
    },
    assemblyOSXCoreTask := Def.sequential(
      Def.task { osxCoreTask }
    ).value
    */
  )

  def coreDependencySettings : Seq[Setting[_]] = {
    Seq(
      libraryDependencies ++= Seq(
      "commons-cli" % "commons-cli" % "1.2",
      "org.rogach" %% "scallop" % "0.9.5",
      "org.scalatest"    %% "scalatest" % "2.2.4" % "test",
      "org.slf4j" % "slf4j-log4j12" % "1.7.5",
      "com.typesafe" % "config" % "1.2.1",
      "colt" % "colt" % "1.2.0",
      "com.twitter" %% "chill" % "0.7.2",
      "org.scalanlp" %% "breeze" % "0.12", 
      "com.typesafe.akka" %% "akka-actor" % "2.4.11",
      "com.typesafe.akka" %% "akka-agent" % "2.4.11",
      versionDependencies(scalaVersion.value)
      )
    )
  }

  def sparkDependencySettings : Seq[Setting[_]] = {
    Seq(
      libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.1.0",
      "org.apache.spark" %% "spark-sql"  % "2.1.0",
      "org.apache.spark" %% "spark-mllib"  % "2.1.0"
      )
    )
  }

  def mxNetDependencySettings : Seq[Setting[_]] = {
    Seq(
      libraryDependencies ++= Seq(
      	// "commons-logging" % "commons-logging" % "1.2"
	// "ml.dmlc" % "xgboost4j" % "0.7",
        // "ml.dmlc.mxnet" % "mxnet-core_2.11" % "0.1.2-SNAPSHOT"
      )
    )
    
  }

  lazy val assemblyLinuxCoreTask = TaskKey[Unit]("linux-core")
  lazy val assemblyOSXCoreTask = TaskKey[Unit]("osx-core")

  val osXJVMLibs =
     (file("mandolin-mx") / "pre-compiled" / "xgboost" / "osx" * "*.jar") +++
     (file("mandolin-mx") / "pre-compiled" / "mxnet" / "osx" * "*.jar")

  val linuxJVMLibs =
     (file("mandolin-mx") / "pre-compiled" / "xgboost" / "linux" * "*.jar") +++
     (file("mandolin-mx") / "pre-compiled" / "mxnet" / "linux-cpu" * "*.jar")

  val linuxJVMLibsGPU =
     (file("mandolin-mx") / "pre-compiled" / "xgboost" / "linux" * "*.jar") +++
     (file("mandolin-mx") / "pre-compiled" / "mxnet" / "linux-gpu" * "*.jar")

  val nonNativeMxOSXLibs =
    (file("mandolin-mx") / "pre-compiled" / "xgboost" / "osx" * "*.jar") +++
    (file("mandolin-mx") / "pre-compiled" / "mxnet" * "*.jar")

  val nonNativeMxLinuxLibs =
    (file("mandolin-mx") / "pre-compiled" / "xgboost" / "linux" * "*.jar") +++
    (file("mandolin-mx") / "pre-compiled" / "mxnet" * "*.jar")

  private def copyFile(destDir: File, file: File) = {
    val fn = file.getName()
    val dstFile = (destDir / fn).toPath
    Files.deleteIfExists(dstFile)
    Files.copy(file.toPath, dstFile)    
  }

  private def setupDistDir() = {
     try { Files.createDirectory(file("dist").toPath) } catch {case _: Throwable => } 
  }

  private def linuxCoreTask = {
    val destDir = file("mandolin-mx") / "lib"
    try { Files.createDirectory(destDir.toPath) } catch {case _: Throwable => }
    // setupDistDir()
    nonNativeMxLinuxLibs.get foreach { f => copyFile(destDir, f) }
  }

  private def osxCoreTask = {
    val destDir = file("mandolin-mx") / "lib"
    try { Files.createDirectory(destDir.toPath) } catch {case _: Throwable => }
    // setupDistDir()
    nonNativeMxOSXLibs.get foreach { f => copyFile(destDir, f) }
  }

  def versionDependencies(v:String) = v match {
    case "2.10.5" => "net.ceedubs" %% "ficus" % "1.0.1"
    case _ => "net.ceedubs" %% "ficus" % "1.1.2"
  }

  def assemblyProjSettings(subProj: String) : Seq[Setting[_]] = Seq(
    test in assembly := {},
    logLevel in assembly := Level.Error,

    assemblyMergeStrategy in assembly := conflictRobustMergeStrategy,
    assemblyJarName in assembly := ("mandolin-"+subProj+"-"+mainVersion+".jar"),
    assemblyOutputPath in assembly := file("dist") / ("mandolin-"+subProj+"-"+mainVersion+".jar")
  )

  def siteSettings : Seq[Setting[_]] = 
    LaikaPlugin.defaults ++ Seq(includeAPI in Laika := true)

  /*
   * Everything below here is hackery related to merging poorly built library dependencies
   */
  val conflictRobustMergeStrategy: String => MergeStrategy = { 
    case "reference.conf" | "rootdoc.txt" =>
      MergeStrategy.concat
    case PathList(ps @ _*) if isReadme(ps.last) || isLicenseFile(ps.last) =>
      MergeStrategy.rename
    case PathList("META-INF", xs @ _*) =>
      (xs map {_.toLowerCase}) match {
        case ("manifest.mf" :: Nil) | ("index.list" :: Nil) | ("dependencies" :: Nil) =>
          MergeStrategy.discard
        case ps @ (x :: xs) if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
          MergeStrategy.discard
        case "plexus" :: xs =>
          MergeStrategy.discard
        case "services" :: xs =>
          MergeStrategy.filterDistinctLines
        case ("spring.schemas" :: Nil) | ("spring.handlers" :: Nil) =>
          MergeStrategy.filterDistinctLines
        case _ => MergeStrategy.first
      }
    case _ => MergeStrategy.first
  }

  private val ReadMe = """(readme)([.]\w+)?$""".r  
  private def isReadme(fileName: String): Boolean =    
    fileName.toLowerCase match { 
      case ReadMe(_, ext) if ext != ".class" => true      
      case _ => false    }

  private val LicenseFile = """(license|licence|notice|copying)([.]\w+)?$""".r  
  private def isLicenseFile(fileName: String): Boolean =    
    fileName.toLowerCase match {      
      case LicenseFile(_, ext) if ext != ".class" => true // DISLIKE      
      case _ => false    }  

}
