package org.mitre.mandolin
/*
 * Copyright (c) 2014-2015 The MITRE Corporation
 */

/**
 * == Configuration Overview ==
 * Provides utility classes for handling configurations. Configurations
 * are handled through the [[https://github.com/typesafehub/config Typesafe Config]] 
 * library. The class [[org.mitre.mandolin.config.ConfigGeneratedCommandOptions]] constructs
 * a set of command-line options that are generated from the current set of 
 * config paths (i.e. keys). This allows each config option to be set in a configuration
 * file or via command-line option.  Note that Mandolin library and application settings are provided
 * in `src/main/resources/reference.conf` and `src/main/resources/application.conf`.
 * Applications may provide a runtime configuration file to override these values via the `--conf`
 * command-line option.  Full details on all the options can be found by looking at these configuration
 * files and the comments therein.  A few specific configuration options are worth mentioning here:
 *
 * === Logging Configuration === 
 * One complication has to due with logging.  The current Mandolin configuration provides three
 * ways to reconfigure logging. By default, logging is controlled in the log4j.properties file on the path
 * (default at src/main/resources/log4j.properties). All logging is directed to the console and to a file
 * `mandolin.log` in the current application root/launch directory. The output file can be overridden via 
 * the `mandolin.log-file` config path either in the provided configuration file or via command-line. The
 * logging level (for all loggers) can be overridden via `mandolin.log-level`.  For full control over
 * the logging, the application should define a separate log4j.properties file and provide this
 * via configuration or command-line at the path `mandolin.log-config-file`. 
 * 
 * === Help and Configuration Display ===
 * Two final useful command-line options are `--X-display` which will print out the JSON/HOCON object
 * with all the current config paths and values (along with comments) and `--help` which prints out 
 * the list of command-line options to override these further at the command-line.  
 * 
 * === Kryo class registration ===
 * Classes that need to be registered for Kryo serialization are held within the MandolinKryoRegistrator package.
 */
package object config {   
}
