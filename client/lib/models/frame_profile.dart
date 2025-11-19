import 'package:flutter/material.dart';
import 'utility_type.dart';

/// Frame profile defining overlay dimensions for different meter types
class FrameProfile {
  final UtilityType utilityType;
  final double aspectRatio; // width / height (e.g., 5.0 for 5:1)
  final double heightPercent; // Height as percentage of screen (0.0 - 1.0)
  final double widthPercent; // Width as percentage of screen (0.0 - 1.0)
  final double borderRadius; // Border radius in pixels
  final Color frameColor; // Frame border color
  final Color overlayColor; // Overlay dimming color

  const FrameProfile({
    required this.utilityType,
    required this.aspectRatio,
    required this.heightPercent,
    required this.widthPercent,
    this.borderRadius = 12.0,
    this.frameColor = Colors.white,
    this.overlayColor = const Color.fromRGBO(0, 0, 0, 0.6),
  });

  /// Get frame profile for a utility type
  static FrameProfile forUtilityType(UtilityType type) {
    switch (type) {
      case UtilityType.gas:
        return const FrameProfile(
          utilityType: UtilityType.gas,
          aspectRatio: 5.0, // 5:1 - very wide and low (mechanical gas meters)
          heightPercent: 0.15, // 15% of screen height - smaller to focus on digits only
          widthPercent: 0.70, // 70% of screen width - narrower to avoid extra text
          borderRadius: 12.0,
        );
      case UtilityType.water:
        return const FrameProfile(
          utilityType: UtilityType.water,
          aspectRatio: 5.0, // 5:1 - similar to gas meters
          heightPercent: 0.15, // 15% of screen height - smaller to focus on digits only
          widthPercent: 0.70, // 70% of screen width - narrower to avoid extra text
          borderRadius: 12.0,
        );
      case UtilityType.electricity:
        return const FrameProfile(
          utilityType: UtilityType.electricity,
          aspectRatio: 2.5, // 2.5:1 - less wide (electronic displays)
          heightPercent: 0.25,
          widthPercent: 0.80,
          borderRadius: 12.0,
        );
    }
  }

  /// Calculate frame dimensions based on screen size
  Size calculateFrameSize(Size screenSize) {
    // Calculate based on height first
    double frameHeight = screenSize.height * heightPercent;
    double frameWidth = frameHeight * aspectRatio;

    // If width exceeds screen, recalculate based on width
    if (frameWidth > screenSize.width * widthPercent) {
      frameWidth = screenSize.width * widthPercent;
      frameHeight = frameWidth / aspectRatio;
    }

    return Size(frameWidth, frameHeight);
  }
}

