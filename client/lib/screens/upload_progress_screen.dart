import 'package:flutter/material.dart';
import '../models/utility_type.dart';
import '../services/api_service.dart';
import 'results_screen.dart';

class UploadProgressScreen extends StatefulWidget {
  final String imagePath;
  final UtilityType utilityType;

  const UploadProgressScreen({
    super.key,
    required this.imagePath,
    required this.utilityType,
  });

  @override
  State<UploadProgressScreen> createState() => _UploadProgressScreenState();
}

class _UploadProgressScreenState extends State<UploadProgressScreen> {
  bool _isUploading = true;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    print('[UploadProgress] Screen initialized');
    print('[UploadProgress] Image path: ${widget.imagePath}');
    print('[UploadProgress] Utility type: ${widget.utilityType.name}');
    _uploadImage();
  }

  Future<void> _uploadImage() async {
    print('[UploadProgress] Starting upload...');
    try {
      print('[UploadProgress] Calling ApiService.uploadImage...');
      final response = await ApiService.uploadImage(
        widget.imagePath,
        widget.utilityType,
      );

      print('[UploadProgress] Upload successful!');
      print('[UploadProgress] Response: ${response.filename}');
      print('[UploadProgress] Reading value: ${response.readingValue}');
      print('[UploadProgress] Normalization status: ${response.normalizationStatus}');
      print('[UploadProgress] Confidence: ${response.confidenceScore}');
      print('[UploadProgress] Has reading: ${response.hasReading}');

      if (mounted) {
        setState(() {
          _isUploading = false;
        });

        print('[UploadProgress] Navigating to ResultsScreen...');
        // Navigate to results screen
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(
              imagePath: widget.imagePath,
              utilityType: widget.utilityType,
              uploadResponse: response,
            ),
          ),
        );
      }
    } catch (e) {
      print('[UploadProgress] ERROR: Upload failed');
      print('[UploadProgress] Error type: ${e.runtimeType}');
      print('[UploadProgress] Error message: $e');
      if (mounted) {
        setState(() {
          _isUploading = false;
          _errorMessage = e.toString().replaceFirst('Exception: ', '');
        });
        print('[UploadProgress] Error message set: $_errorMessage');
      }
    }
  }

  Future<void> _retryUpload() async {
    setState(() {
      _isUploading = true;
      _errorMessage = null;
    });
    await _uploadImage();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.utilityType.name),
        backgroundColor: Colors.black,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_isUploading) ...[
                const CircularProgressIndicator(),
                const SizedBox(height: 24),
                const Text(
                  'Uploading image...',
                  style: TextStyle(fontSize: 18),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Please wait',
                  style: TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ] else if (_errorMessage != null) ...[
                const Icon(Icons.error_outline, size: 64, color: Colors.red),
                const SizedBox(height: 24),
                Text(
                  'Upload Failed',
                  style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16),
                Text(
                  _errorMessage!,
                  style: const TextStyle(fontSize: 16),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 32),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _retryUpload,
                      icon: const Icon(Icons.refresh),
                      label: const Text('Retry'),
                    ),
                    const SizedBox(width: 16),
                    OutlinedButton(
                      onPressed: () => Navigator.pop(context),
                      child: const Text('Back'),
                    ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

