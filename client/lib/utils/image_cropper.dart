import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import '../models/frame_profile.dart';

/// Utility class for cropping images based on frame profile
class ImageCropper {
  /// Crop image to frame area
  /// 
  /// [imageFile] - Original image file
  /// [controller] - Camera controller to get image dimensions
  /// [frameProfile] - Frame profile with dimensions
  /// [screenSize] - Screen size in logical pixels
  /// 
  /// Returns path to cropped image file
  static Future<String> cropToFrame({
    required File imageFile,
    required CameraController controller,
    required FrameProfile frameProfile,
    required Size screenSize,
  }) async {
    try {
      // Read original image
      final imageBytes = await imageFile.readAsBytes();
      var originalImage = img.decodeImage(imageBytes);
      
      if (originalImage == null) {
        throw Exception('Failed to decode image');
      }
      
      // Handle EXIF orientation on iOS - images may be rotated
      originalImage = img.bakeOrientation(originalImage);
      
      print('[ImageCropper] Original image size: ${originalImage.width}x${originalImage.height}');
      print('[ImageCropper] Screen size: ${screenSize.width}x${screenSize.height}');
      
      // Get camera preview size (this is the size of the preview widget, not the actual image)
      final previewSize = controller.value.previewSize;
      if (previewSize == null) {
        throw Exception('Camera preview size not available');
      }
      
      print('[ImageCropper] Preview size: ${previewSize.width}x${previewSize.height}');
      
      // Calculate frame size on screen
      final frameSize = frameProfile.calculateFrameSize(screenSize);
      print('[ImageCropper] Frame size on screen: ${frameSize.width}x${frameSize.height}');
      
      // Calculate frame position on screen (centered)
      final frameLeftScreen = (screenSize.width - frameSize.width) / 2;
      final frameTopScreen = (screenSize.height - frameSize.height) / 2;
      print('[ImageCropper] Frame position on screen: ($frameLeftScreen, $frameTopScreen)');
      
      // CameraPreview uses BoxFit.cover: scales to fill screen while maintaining aspect ratio
      // The preview stream and captured image may have different aspect ratios
      // We need to correctly map what's visible on screen to what's in the captured image
      
      final previewAspectRatio = previewSize.height / previewSize.width;
      final screenAspectRatio = screenSize.height / screenSize.width;
      final imageAspectRatio = originalImage.height / originalImage.width;
      
      print('[ImageCropper] Aspect ratios - Preview: $previewAspectRatio, Screen: $screenAspectRatio, Image: $imageAspectRatio');
      
      // CameraPreview uses AspectRatio widget which scales preview to fit screen
      // Calculate the actual displayed size of preview on screen
      // The preview is scaled to maintain aspect ratio and fill available space
      
      // Calculate scale factor: how the preview is scaled to fit screen
      final scaleX = screenSize.width / previewSize.width;
      final scaleY = screenSize.height / previewSize.height;
      
      // Use the smaller scale to maintain aspect ratio (BoxFit.cover behavior)
      final scale = scaleX < scaleY ? scaleX : scaleY;
      
      // Calculate actual displayed size of preview
      final displayedPreviewWidth = previewSize.width * scale;
      final displayedPreviewHeight = previewSize.height * scale;
      
      // Calculate offset (centering)
      final offsetX = (screenSize.width - displayedPreviewWidth) / 2;
      final offsetY = (screenSize.height - displayedPreviewHeight) / 2;
      
      print('[ImageCropper] Displayed preview size: ${displayedPreviewWidth}x${displayedPreviewHeight}');
      print('[ImageCropper] Preview offset: offsetX=$offsetX, offsetY=$offsetY');
      print('[ImageCropper] Scale factor: $scale');
      
      // Convert screen coordinates to preview coordinates
      // First, adjust for centering offset
      final frameLeftInDisplayedPreview = frameLeftScreen - offsetX;
      final frameTopInDisplayedPreview = frameTopScreen - offsetY;
      
      // Then convert from displayed preview coordinates to actual preview stream coordinates
      // (divide by scale to get coordinates in preview stream pixels)
      final frameLeftInPreview = frameLeftInDisplayedPreview / scale;
      final frameTopInPreview = frameTopInDisplayedPreview / scale;
      
      print('[ImageCropper] Frame in preview stream: ($frameLeftInPreview, $frameTopInPreview)');
      
      // Now map from preview stream coordinates to actual captured image coordinates
      // The captured image may have different dimensions than the preview stream
      // We need to scale based on the ratio between preview stream size and image size
      
      // Calculate scale: how many image pixels per preview stream pixel
      final imageScaleX = originalImage.width / previewSize.width;
      final imageScaleY = originalImage.height / previewSize.height;
      
      print('[ImageCropper] Scale factors (preview stream to image): scaleX=$imageScaleX, scaleY=$imageScaleY');
      
      // Also need to scale frame size from screen to preview, then to image
      final frameWidthInPreview = frameSize.width / scale;
      final frameHeightInPreview = frameSize.height / scale;
      
      // Calculate crop area in actual image coordinates
      final cropX = (frameLeftInPreview * imageScaleX).round();
      final cropY = (frameTopInPreview * imageScaleY).round();
      final cropWidth = (frameWidthInPreview * imageScaleX).round();
      final cropHeight = (frameHeightInPreview * imageScaleY).round();
      
      print('[ImageCropper] Crop area in image: x=$cropX, y=$cropY, w=$cropWidth, h=$cropHeight');
      
      // Ensure crop area is within image bounds
      final x = cropX.clamp(0, originalImage.width).toInt();
      final y = cropY.clamp(0, originalImage.height).toInt();
      final width = cropWidth.clamp(0, originalImage.width - x).toInt();
      final height = cropHeight.clamp(0, originalImage.height - y).toInt();
      
      print('[ImageCropper] Final crop: x=$x, y=$y, w=$width, h=$height');
      
      // Validate crop area
      if (width <= 0 || height <= 0) {
        print('[ImageCropper] Invalid crop area, returning original image');
        return imageFile.path;
      }
      
      // Crop image
      final croppedImage = img.copyCrop(
        originalImage,
        x: x,
        y: y,
        width: width,
        height: height,
      );
      
      print('[ImageCropper] Cropped image size: ${croppedImage.width}x${croppedImage.height}');
      
      // Ensure minimum size for OCR (at least 500px on the smaller side for better OCR accuracy)
      // OCR works better with larger images
      img.Image finalImage = croppedImage;
      const minSize = 500; // Increased from 300 to 500 for better OCR accuracy
      
      if (croppedImage.width < minSize || croppedImage.height < minSize) {
        final scale = minSize / (croppedImage.width < croppedImage.height 
            ? croppedImage.width 
            : croppedImage.height);
        final newWidth = (croppedImage.width * scale).round();
        final newHeight = (croppedImage.height * scale).round();
        print('[ImageCropper] Upscaling from ${croppedImage.width}x${croppedImage.height} to ${newWidth}x${newHeight}');
        finalImage = img.copyResize(
          croppedImage,
          width: newWidth,
          height: newHeight,
          interpolation: img.Interpolation.cubic,
        );
      }
      
      // Save cropped image
      final croppedBytes = img.encodeJpg(finalImage, quality: 95);
      final croppedPath = imageFile.path.replaceAll(
        RegExp(r'\.[^.]+$'),
        '_cropped.jpg',
      );
      final croppedFile = File(croppedPath);
      await croppedFile.writeAsBytes(croppedBytes);
      
      print('[ImageCropper] Saved cropped image to: $croppedPath');
      
      return croppedPath;
    } catch (e, stackTrace) {
      // If cropping fails, return original image path
      print('[ImageCropper] Error cropping image: $e');
      print('[ImageCropper] Stack trace: $stackTrace');
      return imageFile.path;
    }
  }
}

