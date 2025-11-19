import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
import '../models/utility_type.dart';

/// Service for storing and retrieving meter reading history
class ReadingStorage {
  static const String _keyPrefix = 'meter_reading_';

  /// Get the last reading for a specific utility type
  static Future<int?> getLastReading(UtilityType utilityType) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = '$_keyPrefix${utilityType.name.toLowerCase()}';
      final value = prefs.getInt(key);
      return value;
    } catch (e) {
      print('[ReadingStorage] Error getting last reading: $e');
      return null;
    }
  }

  /// Save a new reading for a utility type
  static Future<bool> saveReading(UtilityType utilityType, int reading) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = '$_keyPrefix${utilityType.name.toLowerCase()}';
      return await prefs.setInt(key, reading);
    } catch (e) {
      print('[ReadingStorage] Error saving reading: $e');
      return false;
    }
  }

  /// Get reading history (last N readings) for a utility type
  static Future<List<int>> getReadingHistory(
    UtilityType utilityType, {
    int limit = 10,
  }) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = '${_keyPrefix}history_${utilityType.name.toLowerCase()}';
      final jsonString = prefs.getString(key);
      
      if (jsonString == null) {
        return [];
      }
      
      final List<dynamic> history = json.decode(jsonString);
      final List<int> readings = history.map((e) => e as int).toList();
      
      // Return last N readings
      if (readings.length > limit) {
        return readings.sublist(readings.length - limit);
      }
      return readings;
    } catch (e) {
      print('[ReadingStorage] Error getting reading history: $e');
      return [];
    }
  }

  /// Add a reading to history
  static Future<bool> addToHistory(UtilityType utilityType, int reading) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final key = '${_keyPrefix}history_${utilityType.name.toLowerCase()}';
      
      // Get existing history
      final history = await getReadingHistory(utilityType, limit: 100);
      
      // Add new reading
      history.add(reading);
      
      // Keep only last 100 readings
      final trimmedHistory = history.length > 100
          ? history.sublist(history.length - 100)
          : history;
      
      // Save back
      final jsonString = json.encode(trimmedHistory);
      return await prefs.setString(key, jsonString);
    } catch (e) {
      print('[ReadingStorage] Error adding to history: $e');
      return false;
    }
  }

  /// Clear all readings for a utility type
  static Future<bool> clearReadings(UtilityType utilityType) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final lastKey = '$_keyPrefix${utilityType.name.toLowerCase()}';
      final historyKey = '${_keyPrefix}history_${utilityType.name.toLowerCase()}';
      
      await prefs.remove(lastKey);
      await prefs.remove(historyKey);
      return true;
    } catch (e) {
      print('[ReadingStorage] Error clearing readings: $e');
      return false;
    }
  }
}

