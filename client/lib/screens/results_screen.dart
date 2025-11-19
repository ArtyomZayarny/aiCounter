import 'package:flutter/material.dart';
import 'dart:io';
import '../models/utility_type.dart';
import '../models/api_response.dart';
import '../services/reading_storage.dart';

class ResultsScreen extends StatefulWidget {
  final String imagePath;
  final UtilityType utilityType;
  final UploadImageResponse uploadResponse;

  const ResultsScreen({
    super.key,
    required this.imagePath,
    required this.utilityType,
    required this.uploadResponse,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  int? _previousReading;
  bool _isLoadingHistory = true;

  @override
  void initState() {
    super.initState();
    _loadPreviousReading();
    _saveCurrentReading();
  }

  Future<void> _loadPreviousReading() async {
    final previous = await ReadingStorage.getLastReading(widget.utilityType);
    if (mounted) {
      setState(() {
        _previousReading = previous;
        _isLoadingHistory = false;
      });
    }
  }

  Future<void> _saveCurrentReading() async {
    if (widget.uploadResponse.readingValueInt != null) {
      await ReadingStorage.saveReading(
        widget.utilityType,
        widget.uploadResponse.readingValueInt!,
      );
      await ReadingStorage.addToHistory(
        widget.utilityType,
        widget.uploadResponse.readingValueInt!,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.utilityType.name} - Results'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image preview
            Container(
              height: 200,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  File(widget.imagePath),
                  fit: BoxFit.cover,
                ),
              ),
            ),
            const SizedBox(height: 24),
            // Upload status
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green),
                        const SizedBox(width: 8),
                        Text(
                          'Upload Successful',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    _buildInfoRow('Utility Type', widget.utilityType.name),
                    if (widget.uploadResponse.filename != null)
                      _buildInfoRow('Filename', widget.uploadResponse.filename!),
                    if (widget.uploadResponse.fileSize != null)
                      _buildInfoRow(
                        'File Size',
                        '${(widget.uploadResponse.fileSize! / 1024).toStringAsFixed(2)} KB',
                      ),
                    if (_previousReading != null && !_isLoadingHistory)
                      _buildInfoRow(
                        'Previous Reading',
                        _previousReading.toString().padLeft(8, '0'),
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            // OCR Results
            if (widget.uploadResponse.hasReading)
              Card(
                color: widget.uploadResponse.isHighConfidence
                    ? Colors.green.shade50
                    : Colors.orange.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            widget.uploadResponse.isHighConfidence
                                ? Icons.check_circle
                                : Icons.warning_amber_rounded,
                            color: widget.uploadResponse.isHighConfidence
                                ? Colors.green
                                : Colors.orange,
                            size: 32,
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'Recognized Reading',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      // Large display of recognized reading
                      Center(
                        child: Text(
                          widget.uploadResponse.readingValue ?? 'N/A',
                          style: TextStyle(
                            fontSize: 48,
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context).primaryColor,
                            letterSpacing: 2,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      _buildInfoRow(
                        'Confidence',
                        '${widget.uploadResponse.confidenceScore?.toStringAsFixed(1) ?? 'N/A'}%',
                      ),
                      // Show comparison with previous reading if available
                      if (_previousReading != null && 
                          widget.uploadResponse.readingValueInt != null &&
                          !_isLoadingHistory) ...[
                        const SizedBox(height: 12),
                        _buildComparisonCard(context),
                      ],
                      if (widget.uploadResponse.normalizationStatus != null && 
                          widget.uploadResponse.normalizationStatus != "ok") ...[
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.orange.shade100,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              const Icon(
                                Icons.warning_amber_rounded,
                                color: Colors.orange,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _getStatusMessage(widget.uploadResponse.normalizationStatus!),
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.orange.shade900,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                      if (!widget.uploadResponse.isHighConfidence) ...[
                        const SizedBox(height: 12),
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.orange.shade100,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              const Icon(
                                Icons.info_outline,
                                color: Colors.orange,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  'Low confidence. Please verify the reading and retake if needed.',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.orange.shade900,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
              )
            else
              Card(
                color: Colors.red.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      const Icon(
                        Icons.error_outline,
                        size: 48,
                        color: Colors.red,
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'OCR Recognition Failed',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        widget.uploadResponse.message,
                        textAlign: TextAlign.center,
                        style: const TextStyle(fontSize: 14),
                      ),
                      if (widget.uploadResponse.confidenceScore != null) ...[
                        const SizedBox(height: 12),
                        _buildInfoRow(
                          'Confidence',
                          '${widget.uploadResponse.confidenceScore!.toStringAsFixed(1)}%',
                        ),
                      ],
                      if (widget.uploadResponse.normalizationStatus != null) ...[
                        const SizedBox(height: 8),
                        _buildInfoRow(
                          'Status',
                          widget.uploadResponse.normalizationStatus!,
                        ),
                      ],
                      if (widget.uploadResponse.readingValue != null) ...[
                        const SizedBox(height: 8),
                        _buildInfoRow(
                          'Raw Reading',
                          widget.uploadResponse.readingValue!,
                        ),
                      ],
                      const SizedBox(height: 12),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.red.shade100,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: const Row(
                          children: [
                            Icon(
                              Icons.lightbulb_outline,
                              color: Colors.red,
                              size: 20,
                            ),
                            SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                'Tip: Ensure good lighting, clear focus, and the meter display is fully visible.',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.red,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            const SizedBox(height: 24),
            // Action buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.popUntil(context, (route) => route.isFirst);
                    },
                    icon: const Icon(Icons.home),
                    label: const Text('Home'),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Retake'),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _getStatusMessage(String status) {
    switch (status) {
      case "ok":
        return "Reading recognized successfully";
      case "decreased":
        return "Warning: Reading decreased from previous value. Please verify.";
      case "too_large_jump":
        return "Warning: Reading increased abnormally. Please verify.";
      case "bad_length":
        return "Reading length doesn't match expected format. Please verify.";
      case "no_digits":
        return "No digits found in image. Please retake photo.";
      default:
        return "Status: $status";
    }
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w500,
                color: Colors.grey.shade700,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComparisonCard(BuildContext context) {
    if (_previousReading == null || 
        widget.uploadResponse.readingValueInt == null) {
      return const SizedBox.shrink();
    }

    final current = widget.uploadResponse.readingValueInt!;
    final previous = _previousReading!;
    final difference = current - previous;
    final isIncrease = difference > 0;
    final isDecrease = difference < 0;

    Color cardColor;
    IconData icon;
    String message;

    if (isDecrease) {
      cardColor = Colors.red.shade50;
      icon = Icons.warning;
      message = 'Warning: Reading decreased from previous value';
    } else if (difference > previous * 3) {
      // More than 300% increase
      cardColor = Colors.orange.shade50;
      icon = Icons.info_outline;
      message = 'Large increase detected. Please verify.';
    } else {
      cardColor = Colors.blue.shade50;
      icon = Icons.trending_up;
      message = 'Reading increased normally';
    }

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: cardColor == Colors.red.shade50 
                  ? Colors.red 
                  : cardColor == Colors.orange.shade50 
                      ? Colors.orange 
                      : Colors.blue,
                  size: 20),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  message,
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: cardColor == Colors.red.shade50 
                        ? Colors.red.shade900 
                        : cardColor == Colors.orange.shade50 
                            ? Colors.orange.shade900 
                            : Colors.blue.shade900,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Previous: ${previous.toString().padLeft(8, '0')}',
                style: const TextStyle(fontSize: 12),
              ),
              Text(
                'Difference: ${isIncrease ? '+' : ''}$difference',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: isDecrease ? Colors.red : Colors.green,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

