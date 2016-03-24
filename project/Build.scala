import sbt._
import Keys._
import sbtassembly.Plugin._
import sbtassembly.AssemblyUtils._
import AssemblyKeys._
import laika.sbt.LaikaSbtPlugin.{LaikaPlugin, LaikaKeys}
import LaikaKeys._

object MandolinBuild extends Build {

  lazy val root = Project(id = "mandolin", base = file(".")).
                            settings(rootSettings:_*).
                            aggregate(mandolinCore, mandolinSpark)

  lazy val mandolinCore = Project(id = "mandolin-core", base = file("mandolin-core")).
                            settings(coreSettings:_*).
                            settings(coreDependencySettings:_*).
                            settings(assemblyProjSettings("core"):_*).
                            settings(siteSettings:_*).
                            settings(net.virtualvoid.sbt.graph.Plugin.graphSettings: _*)


  lazy val mandolinSpark = Project(id = "mandolin-spark", base = file("mandolin-spark")).
                            settings(sparkSettings:_*).
                            settings(sparkDependencySettings:_*).
                            settings(assemblyProjSettings("spark"):_*).
                            //settings(siteSettings:_*).
                            settings(net.virtualvoid.sbt.graph.Plugin.graphSettings: _*) dependsOn(mandolinCore)

  def rootSettings = sharedSettings ++ Seq( name := "mandolin" )

  def sharedSettings : Seq[Setting[_]] = Defaults.defaultSettings ++ Seq(
    organization := "org.mitre.mandolin",
    version := "0.3.3-SNAPSHOT",
    scalaVersion := "2.11.7",
    crossScalaVersions := Seq("2.10.5","2.11.7"),
    publishTo := {
       val nexus = "https://oss.sonatype.org/"
       if (isSnapshot.value)
         Some("snapshots" at nexus + "content/repositories/snapshots")
       else
         Some("releases" at nexus + "service/local/staging/deploy/maven2")
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
      versionDependencies(scalaVersion.value)
      )
    )
  }

  def sparkDependencySettings : Seq[Setting[_]] = {
    Seq(
      libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "1.5.2",
      "org.apache.spark" %% "spark-sql"  % "1.5.2",
      "org.apache.spark" %% "spark-mllib"  % "1.5.2"      
      )
    )
  }  

  def versionDependencies(v:String) = v match {
    case "2.10.5" => "net.ceedubs" %% "ficus" % "1.0.1"
    case _ => "net.ceedubs" %% "ficus" % "1.1.2"
  }

  def assemblyProjSettings(subProj: String) : Seq[Setting[_]] = assemblySettings ++ Seq(
    test in assembly := {},
    jarName in assembly := ("mandolin-"+subProj+"-assembly-" + version.value + "_" + scalaVersion.value + ".jar"),
    logLevel in assembly := Level.Error, 
    mergeStrategy in assembly := conflictRobustMergeStrategy,
    mainClass in assembly := Some("org.mitre.mandolin.app.Driver")
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
