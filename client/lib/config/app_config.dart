/// Application configuration
class AppConfig {
  // API Configuration
  // For iOS Simulator/Android Emulator: use 'http://localhost:8000'
  // For physical device: use your computer's IP address
  // You can override this by setting the environment variable FLUTTER_API_BASE_URL
  static String get apiBaseUrl {
    // Try to get from environment variable first
    const envUrl = String.fromEnvironment('API_BASE_URL');
    if (envUrl.isNotEmpty) {
      return envUrl;
    }
    
    // Default to localhost for simulator/emulator
    // Change this to your computer's IP for physical device testing
    return 'http://192.168.101.79:8000';
  }
  
  // API timeout settings
  static const Duration apiTimeout = Duration(seconds: 30);
  static const Duration healthCheckTimeout = Duration(seconds: 5);
  
  // OCR settings
  static const double minConfidenceScore = 60.0;
  static const double highConfidenceThreshold = 80.0;
  
  // Image settings
  static const int minImageSizeForOCR = 300; // pixels
  static const int imageQuality = 95; // JPEG quality (0-100)
}

