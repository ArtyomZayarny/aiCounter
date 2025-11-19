import 'package:flutter/material.dart';
import '../models/frame_profile.dart';

/// Camera overlay widget with framing guide
/// Shows a dimmed overlay with a transparent frame for meter reading area
class CameraOverlay extends StatelessWidget {
  final FrameProfile frameProfile;
  final FrameQuality? quality; // Optional: green/red feedback
  final String? hintText;

  const CameraOverlay({
    super.key,
    required this.frameProfile,
    this.quality,
    this.hintText,
  });

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;
    final frameSize = frameProfile.calculateFrameSize(screenSize);

    return Stack(
      children: [
        // Dimmed overlay with transparent frame (shows camera through the frame)
        _DimmedOverlay(
          frameSize: frameSize,
          screenSize: screenSize,
          overlayColor: frameProfile.overlayColor,
          borderRadius: frameProfile.borderRadius,
        ),
        // Frame border (white border around the transparent area)
        _FrameBorder(
          frameSize: frameSize,
          screenSize: screenSize,
          frameColor: _getFrameColor(),
          borderRadius: frameProfile.borderRadius,
          quality: quality,
        ),
        // Corner indicators (optional, for better alignment)
        _CornerIndicators(
          frameSize: frameSize,
          screenSize: screenSize,
          color: _getFrameColor(),
        ),
        // Hint text (optional)
        if (hintText != null)
          _HintText(
            hintText: hintText!,
            frameSize: frameSize,
            screenSize: screenSize,
          ),
      ],
    );
  }

  Color _getFrameColor() {
    switch (quality) {
      case FrameQuality.good:
        return Colors.green;
      case FrameQuality.poor:
        return Colors.red;
      case FrameQuality.unknown:
      default:
        return frameProfile.frameColor;
    }
  }
}

/// Quality indicator for frame feedback
enum FrameQuality {
  good, // Green - digits detected
  poor, // Red - no digits or poor quality
  unknown, // White - default state
}

/// Dimmed overlay with transparent frame cutout
class _DimmedOverlay extends StatelessWidget {
  final Size frameSize;
  final Size screenSize;
  final Color overlayColor;
  final double borderRadius;

  const _DimmedOverlay({
    required this.frameSize,
    required this.screenSize,
    required this.overlayColor,
    required this.borderRadius,
  });

  @override
  Widget build(BuildContext context) {
    // Calculate frame position (centered)
    final frameLeft = (screenSize.width - frameSize.width) / 2;
    final frameTop = (screenSize.height - frameSize.height) / 2;

    return CustomPaint(
      painter: _OverlayPainter(
        frameRect: RRect.fromRectAndRadius(
          Rect.fromLTWH(
            frameLeft,
            frameTop,
            frameSize.width,
            frameSize.height,
          ),
          Radius.circular(borderRadius),
        ),
        overlayColor: overlayColor,
      ),
      size: screenSize,
    );
  }
}

/// Custom painter for dimmed overlay with transparent frame
class _OverlayPainter extends CustomPainter {
  final RRect frameRect;
  final Color overlayColor;

  _OverlayPainter({
    required this.frameRect,
    required this.overlayColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw overlay in 4 rectangles around the frame (top, bottom, left, right)
    // This creates a transparent "hole" for the frame area
    
    final overlayPaint = Paint()..color = overlayColor;
    
    // Top rectangle
    if (frameRect.top > 0) {
      canvas.drawRect(
        Rect.fromLTWH(0, 0, size.width, frameRect.top),
        overlayPaint,
      );
    }
    
    // Bottom rectangle
    if (frameRect.bottom < size.height) {
      canvas.drawRect(
        Rect.fromLTWH(0, frameRect.bottom, size.width, size.height - frameRect.bottom),
        overlayPaint,
      );
    }
    
    // Left rectangle
    if (frameRect.left > 0) {
      canvas.drawRect(
        Rect.fromLTWH(0, frameRect.top, frameRect.left, frameRect.height),
        overlayPaint,
      );
    }
    
    // Right rectangle
    if (frameRect.right < size.width) {
      canvas.drawRect(
        Rect.fromLTWH(frameRect.right, frameRect.top, size.width - frameRect.right, frameRect.height),
        overlayPaint,
      );
    }
  }

  @override
  bool shouldRepaint(_OverlayPainter oldDelegate) {
    return oldDelegate.frameRect != frameRect ||
        oldDelegate.overlayColor != overlayColor;
  }
}

/// Frame border widget
class _FrameBorder extends StatelessWidget {
  final Size frameSize;
  final Size screenSize;
  final Color frameColor;
  final double borderRadius;
  final FrameQuality? quality;

  const _FrameBorder({
    required this.frameSize,
    required this.screenSize,
    required this.frameColor,
    required this.borderRadius,
    this.quality,
  });

  @override
  Widget build(BuildContext context) {
    final frameLeft = (screenSize.width - frameSize.width) / 2;
    final frameTop = (screenSize.height - frameSize.height) / 2;

    return Positioned(
      left: frameLeft,
      top: frameTop,
      child: Container(
        width: frameSize.width,
        height: frameSize.height,
        decoration: BoxDecoration(
          border: Border.all(
            color: frameColor,
            width: 3.0,
          ),
          borderRadius: BorderRadius.circular(borderRadius),
          // No background color - fully transparent to show camera
        ),
        child: quality == FrameQuality.good
            ? Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(borderRadius),
                  border: Border.all(
                    color: Colors.green.withValues(alpha: 0.3),
                    width: 2.0,
                  ),
                  // No background color - fully transparent
                ),
              )
            : null,
      ),
    );
  }
}

/// Corner indicators for better alignment
class _CornerIndicators extends StatelessWidget {
  final Size frameSize;
  final Size screenSize;
  final Color color;

  const _CornerIndicators({
    required this.frameSize,
    required this.screenSize,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final frameLeft = (screenSize.width - frameSize.width) / 2;
    final frameTop = (screenSize.height - frameSize.height) / 2;
    final cornerLength = 30.0;
    final cornerWidth = 3.0;

    return Stack(
      children: [
        // Top-left corner
        Positioned(
          left: frameLeft,
          top: frameTop,
          child: CustomPaint(
            painter: _CornerPainter(
              color: color,
              cornerLength: cornerLength,
              cornerWidth: cornerWidth,
              position: CornerPosition.topLeft,
            ),
            size: Size(cornerLength, cornerLength),
          ),
        ),
        // Top-right corner
        Positioned(
          left: frameLeft + frameSize.width - cornerLength,
          top: frameTop,
          child: CustomPaint(
            painter: _CornerPainter(
              color: color,
              cornerLength: cornerLength,
              cornerWidth: cornerWidth,
              position: CornerPosition.topRight,
            ),
            size: Size(cornerLength, cornerLength),
          ),
        ),
        // Bottom-left corner
        Positioned(
          left: frameLeft,
          top: frameTop + frameSize.height - cornerLength,
          child: CustomPaint(
            painter: _CornerPainter(
              color: color,
              cornerLength: cornerLength,
              cornerWidth: cornerWidth,
              position: CornerPosition.bottomLeft,
            ),
            size: Size(cornerLength, cornerLength),
          ),
        ),
        // Bottom-right corner
        Positioned(
          left: frameLeft + frameSize.width - cornerLength,
          top: frameTop + frameSize.height - cornerLength,
          child: CustomPaint(
            painter: _CornerPainter(
              color: color,
              cornerLength: cornerLength,
              cornerWidth: cornerWidth,
              position: CornerPosition.bottomRight,
            ),
            size: Size(cornerLength, cornerLength),
          ),
        ),
      ],
    );
  }
}

enum CornerPosition {
  topLeft,
  topRight,
  bottomLeft,
  bottomRight,
}

class _CornerPainter extends CustomPainter {
  final Color color;
  final double cornerLength;
  final double cornerWidth;
  final CornerPosition position;

  _CornerPainter({
    required this.color,
    required this.cornerLength,
    required this.cornerWidth,
    required this.position,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = cornerWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    switch (position) {
      case CornerPosition.topLeft:
        // Horizontal line (top)
        canvas.drawLine(
          Offset(0, cornerWidth / 2),
          Offset(cornerLength, cornerWidth / 2),
          paint,
        );
        // Vertical line (left)
        canvas.drawLine(
          Offset(cornerWidth / 2, 0),
          Offset(cornerWidth / 2, cornerLength),
          paint,
        );
        break;
      case CornerPosition.topRight:
        // Horizontal line (top)
        canvas.drawLine(
          Offset(0, cornerWidth / 2),
          Offset(cornerLength, cornerWidth / 2),
          paint,
        );
        // Vertical line (right)
        canvas.drawLine(
          Offset(cornerLength - cornerWidth / 2, 0),
          Offset(cornerLength - cornerWidth / 2, cornerLength),
          paint,
        );
        break;
      case CornerPosition.bottomLeft:
        // Horizontal line (bottom)
        canvas.drawLine(
          Offset(0, cornerLength - cornerWidth / 2),
          Offset(cornerLength, cornerLength - cornerWidth / 2),
          paint,
        );
        // Vertical line (left)
        canvas.drawLine(
          Offset(cornerWidth / 2, 0),
          Offset(cornerWidth / 2, cornerLength),
          paint,
        );
        break;
      case CornerPosition.bottomRight:
        // Horizontal line (bottom)
        canvas.drawLine(
          Offset(0, cornerLength - cornerWidth / 2),
          Offset(cornerLength, cornerLength - cornerWidth / 2),
          paint,
        );
        // Vertical line (right)
        canvas.drawLine(
          Offset(cornerLength - cornerWidth / 2, 0),
          Offset(cornerLength - cornerWidth / 2, cornerLength),
          paint,
        );
        break;
    }
  }

  @override
  bool shouldRepaint(_CornerPainter oldDelegate) {
    return oldDelegate.color != color ||
        oldDelegate.cornerLength != cornerLength ||
        oldDelegate.cornerWidth != cornerWidth ||
        oldDelegate.position != position;
  }
}

/// Hint text below the frame
class _HintText extends StatelessWidget {
  final String hintText;
  final Size frameSize;
  final Size screenSize;

  const _HintText({
    required this.hintText,
    required this.frameSize,
    required this.screenSize,
  });

  @override
  Widget build(BuildContext context) {
    final frameTop = (screenSize.height - frameSize.height) / 2;
    final hintTop = frameTop + frameSize.height + 20;

    return Positioned(
      left: 20,
      right: 20,
      top: hintTop,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(
          hintText,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
    );
  }
}

