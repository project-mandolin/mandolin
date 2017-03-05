package org.mitre.mandolin.config

/**
 * This trait should be mixed into any main object.
 * It sets up the logger using the current date/time so that 
 * multiple runs don't write to the same log-file
 */
trait LogInit {
  
  val format = new java.text.SimpleDateFormat("yyyy-MM-dd_hhmmss")  
  System.setProperty("current.date", format.format(new java.util.Date))
}