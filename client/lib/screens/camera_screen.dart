import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../models/utility_type.dart';
import '../models/frame_profile.dart';
import '../widgets/camera_overlay.dart';
import '../utils/image_cropper.dart';
import 'upload_progress_screen.dart';

class CameraScreen extends StatefulWidget {
  final UtilityType utilityType;

  const CameraScreen({super.key, required this.utilityType});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isLoading = true;
  String? _errorMessage;
  FrameProfile? _frameProfile;
  bool _isCapturing = false; // Prevent multiple simultaneous captures

  @override
  void initState() {
    super.initState();
    _frameProfile = FrameProfile.forUtilityType(widget.utilityType);
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        setState(() {
          _errorMessage = 'No cameras found';
          _isLoading = false;
        });
        return;
      }

      // Use back camera by default
      final backCamera = _cameras!.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras!.first,
      );

      _controller = CameraController(
        backCamera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _controller!.initialize();

      if (mounted) {
        setState(() {
          _isInitialized = true;
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Camera initialization error: $e';
        _isLoading = false;
      });
    }
  }

  Future<void> _takePicture() async {
    // Prevent multiple simultaneous captures
    if (!_isInitialized || _controller == null || _frameProfile == null || _isCapturing) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      // Take picture
      final XFile image = await _controller!.takePicture();

      if (!mounted) {
        setState(() {
          _isCapturing = false;
        });
        return;
      }

      // Crop image to frame area
      final screenSize = MediaQuery.of(context).size;
      final croppedImagePath = await ImageCropper.cropToFrame(
        imageFile: File(image.path),
        controller: _controller!,
        frameProfile: _frameProfile!,
        screenSize: screenSize,
      );

      // Navigate to upload progress screen with cropped image
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => UploadProgressScreen(
              imagePath: croppedImagePath,
              utilityType: widget.utilityType,
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error taking picture: $e')));
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.utilityType.name)),
        body: const Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.utilityType.name)),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                style: const TextStyle(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Back'),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isInitialized || _controller == null) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.utilityType.name)),
        body: const Center(child: Text('Camera not initialized')),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.utilityType.name),
        backgroundColor: Colors.black,
      ),
      body: Stack(
        children: [
          // Camera preview
          SizedBox.expand(child: CameraPreview(_controller!)),
          // Camera overlay with framing guide
          if (_frameProfile != null)
            CameraOverlay(
              frameProfile: _frameProfile!,
              hintText: 'Align the meter display with the frame',
            ),
          // Utility type indicator
          Positioned(
            top: 16,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(widget.utilityType.icon, color: Colors.white),
                  const SizedBox(width: 8),
                  Text(
                    widget.utilityType.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
          // Capture button
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: FloatingActionButton.large(
                onPressed: _isCapturing ? null : _takePicture,
                backgroundColor: _isCapturing ? Colors.grey : Colors.white,
                child: _isCapturing
                    ? const SizedBox(
                        width: 24,
                        height: 24,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Icon(Icons.camera_alt, color: Colors.black),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
