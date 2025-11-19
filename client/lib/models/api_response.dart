/// Model for API response from image upload
class UploadImageResponse {
  final String status;
  final String message;
  final String? filePath;
  final String? filename;
  final String? utilityType;
  final int? fileSize;
  final String? contentType;
  final String? readingValue; // OCR recognized meter reading (string with leading zeros)
  final int? readingValueInt; // OCR recognized meter reading (integer for calculations)
  final double? confidenceScore; // OCR confidence (0-100)
  final String? rawText; // Raw OCR output for debugging
  final String? normalizationStatus; // Status from normalization (e.g., "ok", "bad_length")

  UploadImageResponse({
    required this.status,
    required this.message,
    this.filePath,
    this.filename,
    this.utilityType,
    this.fileSize,
    this.contentType,
    this.readingValue,
    this.readingValueInt,
    this.confidenceScore,
    this.rawText,
    this.normalizationStatus,
  });

  factory UploadImageResponse.fromJson(Map<String, dynamic> json) {
    return UploadImageResponse(
      status: json['status'] as String,
      message: json['message'] as String,
      filePath: json['file_path'] as String?,
      filename: json['filename'] as String?,
      utilityType: json['utility_type'] as String?,
      fileSize: json['file_size'] as int?,
      contentType: json['content_type'] as String?,
      readingValue: json['reading_value'] as String?,
      readingValueInt: json['reading_value_int'] as int?,
      confidenceScore: json['confidence_score'] != null
          ? (json['confidence_score'] as num).toDouble()
          : null,
      rawText: json['raw_text'] as String?,
      normalizationStatus: json['normalization_status'] as String?,
    );
  }

  /// Check if OCR recognition was successful and normalized
  /// Only accepts readings with status "ok"
  bool get hasReading => readingValue != null && readingValue!.isNotEmpty && normalizationStatus == "ok";

  /// Check if confidence is high enough (>= 80%)
  bool get isHighConfidence =>
      confidenceScore != null && confidenceScore! >= 80.0;
}

