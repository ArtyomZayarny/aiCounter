import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'dart:convert';
import '../models/api_response.dart';
import '../models/utility_type.dart';
import '../config/app_config.dart';

/// API service for communicating with the backend server
class ApiService {
  static String get baseUrl => AppConfig.apiBaseUrl;

  /// Upload image to backend server
  ///
  /// [imagePath] - Path to the image file
  /// [utilityType] - Type of utility meter
  ///
  /// Returns [UploadImageResponse] on success
  /// Throws [Exception] on failure
  static Future<UploadImageResponse> uploadImage(
    String imagePath,
    UtilityType utilityType,
  ) async {
    print('[API] Starting image upload...');
    print('[API] Image path: $imagePath');
    print('[API] Utility type: ${utilityType.name}');
    print('[API] Base URL: $baseUrl');
    
    try {
      final file = File(imagePath);
      if (!await file.exists()) {
        print('[API] ERROR: Image file does not exist at path: $imagePath');
        throw Exception('Image file does not exist');
      }

      final fileSize = await file.length();
      print('[API] File exists, size: $fileSize bytes');

      // Create multipart request
      final url = Uri.parse('$baseUrl/api/upload-image');
      print('[API] Request URL: $url');
      
      final request = http.MultipartRequest('POST', url);

      // Add file with explicit content-type
      print('[API] Adding file to request...');
      
      // Determine content-type from file extension
      final extension = imagePath.toLowerCase().split('.').last;
      MediaType contentType;
      
      if (extension == 'png') {
        contentType = MediaType('image', 'png');
      } else if (extension == 'heic' || extension == 'heif') {
        contentType = MediaType('image', 'heic');
      } else {
        // Default to jpeg for jpg, jpeg, and unknown extensions
        contentType = MediaType('image', 'jpeg');
      }
      
      print('[API] Detected file extension: $extension, using content-type: ${contentType.toString()}');
      
      final multipartFile = await http.MultipartFile.fromPath(
        'file',
        imagePath,
        contentType: contentType,
      );
      request.files.add(multipartFile);
      print('[API] File added, content-type: ${multipartFile.contentType}');

      // Add utility type
      final utilityTypeValue = utilityType.name.toLowerCase();
      request.fields['utility_type'] = utilityTypeValue;
      print('[API] Utility type field added: $utilityTypeValue');
      print('[API] Request fields: ${request.fields}');
      print('[API] Request files count: ${request.files.length}');

      // Send request
      print('[API] Sending request...');
      final streamedResponse = await request.send().timeout(
        AppConfig.apiTimeout,
        onTimeout: () {
          print('[API] ERROR: Request timeout');
          throw Exception('Request timeout. Please check your connection.');
        },
      );

      print('[API] Response received, status code: ${streamedResponse.statusCode}');
      print('[API] Response headers: ${streamedResponse.headers}');

      // Get response
      final response = await http.Response.fromStream(streamedResponse);
      print('[API] Response body: ${response.body}');

      if (response.statusCode == 200) {
        print('[API] SUCCESS: Upload successful');
        final jsonResponse = json.decode(response.body) as Map<String, dynamic>;
        return UploadImageResponse.fromJson(jsonResponse);
      } else {
        print('[API] ERROR: Server returned status ${response.statusCode}');
        print('[API] Response body: ${response.body}');
        // Try to parse error message
        try {
          final errorJson = json.decode(response.body) as Map<String, dynamic>;
          final errorMessage = errorJson['detail'] as String? ?? 'Unknown error';
          print('[API] Parsed error message: $errorMessage');
          throw Exception('Server error: $errorMessage');
        } catch (e) {
          if (e is Exception && e.toString().contains('Server error:')) {
            rethrow;
          }
          print('[API] Could not parse error JSON, using status code');
          throw Exception(
            'Server error: ${response.statusCode} - ${response.reasonPhrase}',
          );
        }
      }
    } on SocketException catch (e) {
      print('[API] ERROR: SocketException - ${e.message}');
      throw Exception(
        'Network error. Please check your connection and ensure the server is running.',
      );
    } on HttpException catch (e) {
      print('[API] ERROR: HttpException - ${e.message}');
      throw Exception('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      print('[API] ERROR: FormatException - ${e.message}');
      throw Exception('Invalid response from server');
    } catch (e) {
      print('[API] ERROR: Unexpected error - $e');
      print('[API] Error type: ${e.runtimeType}');
      if (e is Exception) {
        rethrow;
      }
      throw Exception('Unexpected error: $e');
    }
  }

  /// Check if server is available
  static Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(AppConfig.healthCheckTimeout);
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
  
  /// Get the current API base URL (for debugging)
  static String getApiBaseUrl() => baseUrl;
}

