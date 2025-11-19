"""
OCR processing module for utility meter reading recognition.
Uses PaddleOCR (primary), EasyOCR (fallback), and Tesseract OCR (fallback 2) for digit recognition.
Includes image normalization (deskew, contrast enhancement, adaptive binarization) and post-processing validation.
"""
import logging
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

import cv2
import easyocr
import numpy as np
import pytesseract
from PIL import Image

if TYPE_CHECKING:
    from app.models.meter_profile import MeterProfile

logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor for recognizing digits from utility meter images."""

    def __init__(self):
        """Initialize OCR processor with PaddleOCR (primary), EasyOCR (fallback), and Tesseract (fallback 2)."""
        # Initialize PaddleOCR (best for digits) - PRIMARY
        self.paddleocr_reader = None
        try:
            from paddleocr import PaddleOCR
            # Use English model, enable digits only
            self.paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=False,
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully (PRIMARY)")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}. Will use EasyOCR.")
            self.paddleocr_reader = None
        
        # Initialize EasyOCR reader (English + digits only) - FALLBACK
        logger.info("Initializing EasyOCR reader...")
        try:
            # Fix SSL certificate issues on Mac
            import certifi
            import os
            import ssl
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
            
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully (FALLBACK)")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}. Will use Tesseract only.")
            self.easyocr_reader = None
        
        # Initialize Tesseract as fallback
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract found at: {tesseract_path}")
        else:
            logger.warning("Tesseract not found in PATH, using default")
        
        # Configure Tesseract for digit-only recognition
        self.config = "--psm 8 -c tessedit_char_whitelist=0123456789"
        
        # Initialize YOLOv8 for meter frame detection (optional, falls back to contour detection)
        self.yolo_model = None
        try:
            from ultralytics import YOLO
            import os
            
            # Check for custom model path (environment variable or default location)
            custom_model_path = os.getenv('YOLO_METER_MODEL_PATH')
            if custom_model_path and Path(custom_model_path).exists():
                try:
                    self.yolo_model = YOLO(custom_model_path)
                    logger.info(f"YOLOv8 initialized with custom model: {custom_model_path}")
                except Exception as e:
                    logger.warning(f"Could not load custom YOLOv8 model: {e}. Trying pre-trained model.")
                    custom_model_path = None
            
            # Fallback to pre-trained model if custom model not available
            if self.yolo_model is None:
                try:
                    self.yolo_model = YOLO('yolov8n.pt')  # Pre-trained model
                    logger.info("YOLOv8 initialized (using pre-trained model)")
                    logger.info("Note: For better accuracy, train a custom model on meter images.")
                    logger.info("See docs/YOLO_TRAINING.md for instructions.")
                except Exception as e:
                    logger.warning(f"Could not load YOLOv8 model: {e}. Will use contour-based detection.")
                    self.yolo_model = None
        except (ImportError, AttributeError) as e:
            logger.warning(f"ultralytics not available or incompatible: {e}. Will use contour-based detection only.")
            self.yolo_model = None
        
        logger.info("OCR Processor initialized")

    def _find_meter_display_region(self, gray: np.ndarray, original_img: np.ndarray = None) -> tuple:
        """
        Find the region containing meter reading display using YOLOv8 (if available) 
        or contour-based detection (fallback).
        
        Args:
            gray: Grayscale image
            original_img: Original color image (for YOLO detection, optional)
            
        Returns:
            Tuple of (x, y, w, h) for the display region, or None if not found
        """
        # Try YOLOv8 detection first (if available)
        if self.yolo_model is not None and original_img is not None:
            try:
                return self._find_meter_region_with_yolo(original_img)
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}. Falling back to contour detection.")
        
        # Fallback to contour-based detection
        return self._find_meter_region_with_contours(gray)
    
    def _find_meter_region_with_yolo(self, img: np.ndarray) -> tuple:
        """
        Find meter display region using YOLOv8 object detection.
        
        Note: This uses a pre-trained YOLOv8 model. For best results, you should:
        1. Collect 40-300 images of your specific meter types
        2. Label them with bounding boxes around the display area
        3. Fine-tune YOLOv8 on your dataset
        
        Args:
            img: Color image (BGR format from OpenCV)
            
        Returns:
            Tuple of (x, y, w, h) for the display region, or None if not found
        """
        try:
            # Run YOLO detection
            results = self.yolo_model(img, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            # Get detections from first result
            result = results[0]
            
            # For now, we'll look for the largest detected object in the upper part
            # In a custom-trained model, you'd have a specific class for "meter_display"
            boxes = result.boxes
            
            if boxes is None or len(boxes) == 0:
                return None
            
            height, width = img.shape[:2]
            
            # Filter boxes in upper 60% of image (where readings usually are)
            valid_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_y = (y1 + y2) / 2
                
                # Prefer boxes in upper part of image
                if center_y < height * 0.6:
                    valid_boxes.append((x1, y1, x2, y2, box.conf[0].cpu().numpy()))
            
            if not valid_boxes:
                return None
            
            # Select the box with highest confidence in upper region
            valid_boxes.sort(key=lambda b: b[4], reverse=True)
            x1, y1, x2, y2, conf = valid_boxes[0]
            
            # Convert to (x, y, w, h) format
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            logger.info(f"YOLO detected meter region: {w}x{h} at ({x}, {y}) with confidence {conf:.2f}")
            
            return (x, y, w, h)
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}", exc_info=True)
            return None
    
    def _find_meter_region_with_contours(self, gray: np.ndarray) -> tuple:
        """
        Find meter display region using contour-based detection (fallback method).
        Focuses on the upper/center part of the image where readings are typically located.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (x, y, w, h) for the display region, or None if not found
        """
        try:
            height, width = gray.shape
            
            # Focus on upper 40% of image (readings are almost always at the top)
            # More aggressive cropping to avoid serial numbers and dates at bottom
            search_height = int(height * 0.4)
            search_region = gray[0:search_height, :]
            
            # Apply threshold to find digit regions
            _, thresh = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours - look for rectangular regions (digit cells)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by aspect ratio and size (digit cells are usually rectangular)
            digit_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Digit cells are usually:
                # - Not too small (at least 15x15 pixels)
                # - Not too large (not more than 25% of image width)
                # - Rectangular (aspect ratio between 0.4 and 2.5)
                if (area > 200 and 
                    w > 15 and h > 15 and
                    w < width * 0.25 and h < search_height * 0.25 and
                    0.4 < aspect_ratio < 2.5):
                    digit_contours.append((x, y, w, h, area))
            
            if not digit_contours:
                return None
            
            # Group nearby contours (digit cells are usually close together horizontally)
            # Sort by x coordinate (left to right)
            digit_contours.sort(key=lambda c: c[0])
            
            # Find the largest group of horizontally aligned contours
            groups = []
            current_group = [digit_contours[0]]
            
            for i in range(1, len(digit_contours)):
                prev_x, prev_y, prev_w, prev_h, _ = digit_contours[i-1]
                curr_x, curr_y, curr_w, curr_h, _ = digit_contours[i]
                
                # Check if contours are on similar y-level (horizontally aligned)
                y_diff = abs(curr_y - prev_y)
                x_gap = curr_x - (prev_x + prev_w)
                avg_height = (prev_h + curr_h) / 2
                avg_width = (prev_w + curr_w) / 2
                
                # Contours are aligned if:
                # - Y difference is less than 1.5x average height
                # - X gap is less than 3x average width (digits are close)
                if y_diff < avg_height * 1.5 and x_gap < avg_width * 3:
                    current_group.append(digit_contours[i])
                else:
                    if len(current_group) >= 4:  # At least 4 digits for a reading
                        groups.append(current_group)
                    current_group = [digit_contours[i]]
            
            if len(current_group) >= 4:
                groups.append(current_group)
            
            if not groups:
                return None
            
            # Find the group with the most digits (likely the main reading)
            # Prefer groups in the center/upper part of the image
            def group_score(group):
                num_digits = len(group)
                # Calculate average y position (lower is better - readings are usually at top)
                avg_y = sum(c[1] for c in group) / len(group)
                # Prefer groups higher up (lower y value) and with more digits
                return num_digits * 10 - (avg_y / search_height) * 5
            
            best_group = max(groups, key=group_score)
            
            # Calculate bounding box for the group
            min_x = min(c[0] for c in best_group)
            min_y = min(c[1] for c in best_group)
            max_x = max(c[0] + c[2] for c in best_group)
            max_y = max(c[1] + c[3] for c in best_group)
            
            # Add padding
            padding = 40
            x = max(0, min_x - padding)
            y = max(0, min_y - padding)
            w = min(width - x, max_x - min_x + 2 * padding)
            h = min(search_height - y, max_y - min_y + 2 * padding)
            
            # Validate coordinates
            if x >= width or y >= height or w <= 0 or h <= 0:
                logger.warning(f"Invalid coordinates: x={x}, y={y}, w={w}, h={h}, image={width}x{height}")
                return None
            
            logger.info(f"Found meter display region: {w}x{h} at ({x}, {y}) with {len(best_group)} digit cells")
            return (x, y, w, h)
            
        except Exception as e:
            logger.warning(f"Failed to find meter display region: {e}", exc_info=True)
            return None

    def _preprocess_image(self, image_path: Path) -> Image.Image:
        """
        Preprocess image for better OCR recognition.
        Focuses on finding and extracting the meter reading display area.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed PIL Image
        """
        # Load image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        logger.info(f"Original image shape: {img.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize if image is too large (OCR works better with reasonable sizes)
        # But keep it large enough for good quality
        height, width = gray.shape
        max_dimension = 3000
        min_dimension = 500
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image to: {new_width}x{new_height}")
        elif min(height, width) < min_dimension:
            scale = min_dimension / min(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Upscaled image to: {new_width}x{new_height}")
        
        # Try to find the meter display region using YOLO (if available) or contours
        display_region = self._find_meter_display_region(gray, original_img=img)
        if display_region:
            x, y, w, h = display_region
            gray = gray[y:y+h, x:x+w]
            logger.info(f"Cropped to meter display region: {w}x{h}")
        
        # 2. NORMALIZATION: Deskew (straighten image)
        gray = self._deskew_image(gray)
        
        # 3. NORMALIZATION: Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 4. NORMALIZATION: Adaptive binarization (better than OTSU for uneven lighting)
        # Use adaptive threshold instead of global threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. NORMALIZATION: Noise reduction
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Convert back to PIL Image
        pil_image = Image.fromarray(thresh)
        logger.info(f"Image preprocessed: {pil_image.size}, mode: {pil_image.mode}")
        
        return pil_image
    
    def _deskew_image(self, gray: np.ndarray) -> np.ndarray:
        """
        Straighten image (deskew).
        Corrects meter tilt in photos.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Deskewed grayscale image
        """
        try:
            # Find all non-zero pixels (text)
            coords = np.column_stack(np.where(gray > 0))
            
            if len(coords) < 10:  # Not enough data for deskew
                return gray
            
            # Find tilt angle
            angle = cv2.minAreaRect(coords)[-1]
            
            # Correct angle
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # If angle is small (< 1 degree), don't rotate
            if abs(angle) < 1.0:
                return gray
            
            # Rotate image
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                gray, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.info(f"Deskewed image by {angle:.2f} degrees")
            return rotated
            
        except Exception as e:
            logger.warning(f"Deskew failed: {e}, using original image")
            return gray

    def _recognize_with_paddleocr(
        self, image_path: Path
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Recognize digits using PaddleOCR (best for digits).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (reading_value, confidence_score, raw_text)
        """
        if self.paddleocr_reader is None:
            return None, 0.0, None
        
        try:
            logger.info("Trying PaddleOCR recognition...")
            
            # Read image and try to crop to meter display region first
            img = cv2.imread(str(image_path))
            if img is None:
                return None, 0.0, None
            
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            display_region = self._find_meter_display_region(gray, original_img=img)
            
            if display_region:
                x, y, w, h = display_region
                cropped_img = img[y:y+h, x:x+w]
                logger.info(f"Using cropped region for PaddleOCR: {w}x{h} at ({x}, {y})")
                # Save cropped image temporarily for PaddleOCR
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, cropped_img)
                    img_path_for_ocr = tmp_file.name
            else:
                # Aggressively crop to upper 30% if no region detected
                crop_height = int(height * 0.3)
                cropped_img = img[0:crop_height, :]
                logger.info(f"No region detected, using aggressive crop: top {crop_height}px (30% of image)")
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, cropped_img)
                    img_path_for_ocr = tmp_file.name
            
            # PaddleOCR works with image paths or numpy arrays
            result = self.paddleocr_reader.ocr(img_path_for_ocr, cls=True)
            
            # Clean up temp file
            try:
                import os
                os.unlink(img_path_for_ocr)
            except:
                pass
            
            if not result or not result[0]:
                logger.info("PaddleOCR found no text")
                return None, 0.0, None
            
            # Extract all detected text and confidence scores
            detections = []
            height, width = 0, 0
            
            for line in result[0]:
                if not line:
                    continue
                
                bbox, (text, confidence) = line
                digits = re.sub(r"[^0-9]", "", text)
                
                if digits:
                    # Calculate center position
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    center_x = sum(x_coords) / len(x_coords)
                    center_y = sum(y_coords) / len(y_coords)
                    
                    if width == 0:
                        width = int(max(x_coords))
                        height = int(max(y_coords))
                    
                    norm_x = center_x / width if width > 0 else 0.5
                    norm_y = center_y / height if height > 0 else 0.5
                    
                    detections.append({
                        'text': digits,
                        'confidence': confidence * 100,
                        'length': len(digits),
                        'y': norm_y,
                        'raw_text': text
                    })
            
            if not detections:
                return None, 0.0, None
            
            # Simple selection: prefer 6-8 digit sequences in upper region, starting with 0
            best_sequence = None
            best_confidence = 0.0
            best_score = 0.0
            
            for det in detections:
                length = det['length']
                conf = det['confidence']
                pos_y = det['y']
                text = det['text']
                
                # Skip years
                if length == 4 and 1900 <= int(text) <= 2100:
                    continue
                
                # Score: prefer 6-8 digits, upper region, starting with 0
                if 6 <= length <= 8:
                    length_score = 1.5
                elif length == 5 or length == 9:
                    length_score = 1.0
                else:
                    length_score = 0.5
                
                position_score = 2.0 if pos_y < 0.3 else (1.0 if pos_y < 0.5 else 0.3)
                zero_bonus = 1.3 if text.startswith('0') else 1.0
                
                score = conf * length_score * position_score * zero_bonus
                
                if score > best_score and 5 <= length <= 10:
                    best_sequence = text
                    best_confidence = conf
                    best_score = score
            
            # Try simple combination if no good single detection
            if not best_sequence or len(best_sequence) < 6:
                detections_sorted = sorted(detections, key=lambda d: (d['y'], d['x'] if 'x' in d else 0.5))
                combined = ''.join([d['text'] for d in detections_sorted if d['y'] < 0.5])
                
                # Look for 6-8 digit sequence starting with 0
                for length in [8, 7, 6]:
                    for start in range(len(combined) - length + 1):
                        candidate = combined[start:start + length]
                        if candidate.startswith('0') and candidate.isdigit():
                            avg_conf = sum([d['confidence'] for d in detections_sorted if d['text'] in candidate]) / max(1, len([d for d in detections_sorted if d['text'] in candidate]))
                            if avg_conf > best_confidence:
                                best_sequence = candidate
                                best_confidence = avg_conf
                                break
                    if best_sequence:
                        break
            
            combined_raw_text = " ".join([d['raw_text'] for d in detections])
            
            if best_sequence and 5 <= len(best_sequence) <= 10:
                return best_sequence, round(best_confidence, 2), combined_raw_text
            
            return None, 0.0, combined_raw_text if combined_raw_text else None
            
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {str(e)}", exc_info=True)
            return None, 0.0, None

    def _recognize_with_easyocr(
        self, image_path: Path
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Recognize digits using EasyOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (reading_value, confidence_score, raw_text)
        """
        if self.easyocr_reader is None:
            return None, 0.0, None
        
        try:
            logger.info("Trying EasyOCR recognition...")
            
            # Read image with OpenCV (EasyOCR works with numpy arrays)
            img = cv2.imread(str(image_path))
            if img is None:
                return None, 0.0, None
            
            # Try to crop to meter display region first (if possible)
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            display_region = self._find_meter_display_region(gray, original_img=img)
            
            if display_region:
                x, y, w, h = display_region
                # Crop image to display region for better focus
                cropped_img = img[y:y+h, x:x+w]
                logger.info(f"Using cropped region for EasyOCR: {w}x{h} at ({x}, {y})")
                img_for_ocr = cropped_img
            else:
                # If no region detected, VERY aggressively crop to upper 30% of image
                # Meter readings are almost always in the very top part
                crop_height = int(height * 0.3)
                img_for_ocr = img[0:crop_height, :]
                logger.info(f"No region detected, using very aggressive crop: top {crop_height}px (30% of image)")
            
            # EasyOCR recognition
            results = self.easyocr_reader.readtext(img_for_ocr, allowlist='0123456789')
            
            if not results:
                logger.info("EasyOCR found no text")
                return None, 0.0, None
            
            # Extract all detected text and confidence scores
            detections = []
            
            for (bbox, text, confidence) in results:
                digits = re.sub(r"[^0-9]", "", text)
                if digits:
                    if display_region:
                        x, y, _, _ = display_region
                        center_x = (bbox[0][0] + bbox[2][0]) / 2 + x
                        center_y = (bbox[0][1] + bbox[2][1]) / 2 + y
                    else:
                        center_x = (bbox[0][0] + bbox[2][0]) / 2
                        center_y = (bbox[0][1] + bbox[2][1]) / 2
                    
                    norm_x = center_x / width
                    norm_y = center_y / height
                    
                    detections.append({
                        'text': digits,
                        'confidence': confidence * 100,
                        'length': len(digits),
                        'y': norm_y,
                        'raw_text': text
                    })
            
            if not detections:
                return None, 0.0, None
            
            # Simple selection: prefer 6-8 digit sequences in upper region, starting with 0
            best_sequence = None
            best_confidence = 0.0
            best_score = 0.0
            
            for det in detections:
                length = det['length']
                conf = det['confidence']
                pos_y = det['y']
                text = det['text']
                
                # Skip years
                if length == 4 and 1900 <= int(text) <= 2100:
                    continue
                
                # Score: prefer 6-8 digits, upper region, starting with 0
                if 6 <= length <= 8:
                    length_score = 1.5
                elif length == 5 or length == 9:
                    length_score = 1.0
                else:
                    length_score = 0.5
                
                position_score = 2.0 if pos_y < 0.3 else (1.0 if pos_y < 0.5 else 0.3)
                zero_bonus = 1.3 if text.startswith('0') else 1.0
                
                score = conf * length_score * position_score * zero_bonus
                
                if score > best_score and 5 <= length <= 10:
                    best_sequence = text
                    best_confidence = conf
                    best_score = score
            
            # Try simple combination if no good single detection
            if not best_sequence or len(best_sequence) < 6:
                detections_sorted = sorted(detections, key=lambda d: (d['y'], d.get('x', 0.5)))
                combined = ''.join([d['text'] for d in detections_sorted if d['y'] < 0.5])
                
                # Look for 6-8 digit sequence starting with 0
                for length in [8, 7, 6]:
                    for start in range(len(combined) - length + 1):
                        candidate = combined[start:start + length]
                        if candidate.startswith('0') and candidate.isdigit():
                            avg_conf = sum([d['confidence'] for d in detections_sorted if d['text'] in candidate]) / max(1, len([d for d in detections_sorted if d['text'] in candidate]))
                            if avg_conf > best_confidence:
                                best_sequence = candidate
                                best_confidence = avg_conf
                                break
                    if best_sequence:
                        break
            
            combined_raw_text = " ".join([d['raw_text'] for d in detections])
            
            if best_sequence and 5 <= len(best_sequence) <= 10:
                return best_sequence, round(best_confidence, 2), combined_raw_text
            
            return None, 0.0, combined_raw_text if combined_raw_text else None
            
        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {str(e)}", exc_info=True)
            return None, 0.0, None

    def _post_process_reading(
        self, reading_value: Optional[int], utility_type: Optional[str] = None
    ) -> Optional[int]:
        """
        Post-process and validate recognized reading.
        
        Args:
            reading_value: Recognized reading value
            utility_type: Type of utility meter (gas, water, electricity)
            
        Returns:
            Validated reading value or None if validation failed
        """
        if reading_value is None:
            return None
        
        reading_str = str(reading_value)
        length = len(reading_str)
        
        # 1. Length validation (5-8 digits for meters - some meters have 8 digits)
        if length < 5 or length > 8:
            logger.warning(f"Invalid reading length: {length} digits (expected 5-8)")
            return None
        
        # 2. Filter years (2012, 2021, etc.)
        if length == 4 and 1900 <= reading_value <= 2100:
            logger.warning(f"Reading looks like a year: {reading_value}")
            return None
        
        # 3. Filter 7-digit numbers starting with year
        if length == 7 and reading_str.startswith('20') and int(reading_str[:4]) >= 2000:
            logger.warning(f"Reading looks like year + digits: {reading_value}")
            return None
        
        # 4. Regex validation (digits only - already checked, but just in case)
        if not reading_str.isdigit():
            logger.warning(f"Reading contains non-digit characters: {reading_str}")
            return None
        
        # 5. Reasonable value check (can be extended)
        # For example, meters usually don't show very large numbers
        # But this depends on meter type
        
        logger.info(f"Post-processing validation passed: {reading_value}")
        return reading_value

    def recognize_digits(
        self, 
        image_path: Path, 
        min_confidence: int = 60, 
        utility_type: Optional[str] = None,
        profile: Optional['MeterProfile'] = None,
        prev_value: Optional[int] = None,
    ) -> Tuple[Optional[str], float, Optional[str], str]:
        """
        Recognize digits from a meter image.

        Args:
            image_path: Path to the image file
            min_confidence: Minimum confidence score threshold (0-100)
            utility_type: Type of utility meter (gas, water, electricity)
            profile: Meter profile for normalization (if None, uses default for utility_type)
            prev_value: Previous reading value for validation

        Returns:
            Tuple of (reading_value_str, confidence_score, raw_text, status)
            - reading_value_str: Recognized reading as string with leading zeros (e.g., "01814511") or None
            - confidence_score: OCR confidence (0-100)
            - raw_text: Raw OCR output text (for debugging)
            - status: Normalization status ("ok", "no_digits", "bad_length", "decreased", "too_large_jump")
        """
        try:
            logger.info(f"Starting OCR recognition for image: {image_path}")

            # Load image
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None, 0.0, None, "file_not_found"
            
            # Get or create profile
            if profile is None:
                from app.models.meter_profile import get_profile
                profile = get_profile(utility_type or "gas")
                logger.info(f"Using default profile for {utility_type}: {profile.name}")

            # Collect OCR results - use the best sequence directly, not raw text
            best_reading_value = None
            best_confidence = 0.0
            best_raw_text = None
            all_raw_texts = []
            
            # Try PaddleOCR first (best for digits)
            if self.paddleocr_reader is not None:
                paddleocr_result = self._recognize_with_paddleocr(image_path)
                if paddleocr_result[0] is not None:
                    reading_value, confidence, raw_text = paddleocr_result
                    all_raw_texts.append(raw_text or "")
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_reading_value = reading_value
                        best_raw_text = raw_text
            
            # Try EasyOCR as fallback
            if self.easyocr_reader is not None and best_reading_value is None:
                easyocr_result = self._recognize_with_easyocr(image_path)
                if easyocr_result[0] is not None:
                    reading_value, confidence, raw_text = easyocr_result
                    all_raw_texts.append(raw_text or "")
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_reading_value = reading_value
                        best_raw_text = raw_text
            
            # Try Tesseract as last fallback
            if best_reading_value is None:
                processed_image = self._preprocess_image(image_path)
                tesseract_result = self._recognize_with_tesseract(processed_image, image_path)
                if tesseract_result[0] is not None:
                    reading_value, confidence, raw_text = tesseract_result
                    all_raw_texts.append(raw_text or "")
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_reading_value = reading_value
                        best_raw_text = raw_text
            
            combined_raw_text = " ".join(all_raw_texts) if all_raw_texts else (best_raw_text or "")
            
            # If we have a reading value from OCR, use it directly with simple normalization
            if best_reading_value is not None:
                # best_reading_value is already a string (preserves leading zeros)
                reading_str = best_reading_value
                original_length = len(reading_str)
                
                # Handle length differences
                if len(reading_str) < profile.total_digits:
                    # If reading is 1 digit shorter and starts with 0, accept as-is
                    # (e.g., "0181451" is acceptable for 8-digit format)
                    if len(reading_str) == profile.total_digits - 1 and reading_str.startswith('0'):
                        # Accept 7-digit reading if it's close to expected (8 digits)
                        # This handles cases where last digit is not visible or not needed
                        logger.info(f"Accepting {original_length}-digit reading (close to expected {profile.total_digits}): {reading_str}")
                        # Don't pad - use as-is (user confirmed this is a good result)
                    # If reading is much shorter and starts with 0, pad with leading zeros
                    elif reading_str.startswith('0') and profile.allow_leading_zero:
                        reading_str = reading_str.zfill(profile.total_digits)
                        logger.info(f"Padded reading with leading zeros: {str(best_reading_value)} -> {reading_str}")
                    # For other cases, pad with leading zeros
                    else:
                        if profile.allow_leading_zero:
                            reading_str = reading_str.zfill(profile.total_digits)
                            logger.info(f"Padded reading: {str(best_reading_value)} -> {reading_str}")
                elif len(reading_str) > profile.total_digits:
                    # If too long, take first N digits (but preserve leading zero if present)
                    if reading_str.startswith('0'):
                        # Keep leading zero and take first N digits
                        reading_str = reading_str[:profile.total_digits]
                    else:
                        # Take last N digits
                        reading_str = reading_str[-profile.total_digits:]
                    logger.info(f"Trimmed reading to {profile.total_digits} digits: {reading_str}")
                
                # Accept if length is close to expected (within 1 digit)
                if abs(len(reading_str) - profile.total_digits) <= 1:
                    # Validate against previous reading if available
                    try:
                        reading_int = int(reading_str)
                        status = "ok"
                        
                        if prev_value is not None:
                            if reading_int < prev_value:
                                status = "decreased"
                                logger.warning(f"Reading decreased: {reading_int} < {prev_value}")
                            elif prev_value > 0:
                                jump_percent = ((reading_int - prev_value) / prev_value) * 100
                                if jump_percent > profile.max_jump_percent:
                                    status = "too_large_jump"
                                    logger.warning(f"Large jump: {jump_percent:.1f}%")
                        
                        logger.info(f"Using OCR result directly: {reading_str} (confidence: {best_confidence:.1f}%, status: {status})")
                        return reading_str, best_confidence, combined_raw_text, status
                    except ValueError:
                        logger.warning(f"Could not convert reading to int: {reading_str}")
                        # Fall through to normalizer
                else:
                    logger.warning(f"Reading length too different: got {len(reading_str)}, expected {profile.total_digits}")
                    # Fall through to normalizer as fallback
            
            # Fallback: use normalizer if OCR didn't give us a good result
            if not combined_raw_text:
                logger.warning("No OCR results found")
                return None, 0.0, None, "no_digits"
            
            # Normalize using the normalizer as fallback
            from app.ocr.normalizer import normalize_ocr
            normalized_reading, status = normalize_ocr(
                combined_raw_text,
                profile,
                prev_value
            )
            
            if normalized_reading:
                logger.info(f"Normalized reading (fallback): {normalized_reading} (status: {status}, confidence: {best_confidence:.1f}%)")
                return normalized_reading, best_confidence, combined_raw_text, status
            else:
                logger.warning(f"Normalization failed: {status}")
                return None, best_confidence, combined_raw_text, status

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}", exc_info=True)
            return None, 0.0, None, "error"
    
    def _recognize_with_tesseract(self, processed_image: Image.Image, image_path: Path) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Recognize digits using Tesseract OCR (fallback method).
        Returns string sequence to preserve leading zeros.
        
        Returns:
            Tuple of (reading_sequence, confidence, raw_text)
        """
        try:
            # Try multiple PSM modes
            psm_modes = [
                ("7", "Single text line"),
                ("8", "Single word"),
                ("6", "Single uniform block of text"),
            ]
            
            best_raw_text = None
            best_sequence = None
            best_confidence = 0.0
            
            for psm_mode, description in psm_modes:
                config = f"--psm {psm_mode} -c tessedit_char_whitelist=0123456789"
                try:
                    raw_text = pytesseract.image_to_string(processed_image, config=config)
                    raw_text = raw_text.strip()
                    
                    if raw_text:
                        # Extract digits only
                        digits = re.sub(r"[^0-9]", "", raw_text)
                        
                        # Get confidence
                        ocr_data = pytesseract.image_to_data(
                            processed_image, config=config, output_type=pytesseract.Output.DICT
                        )
                        confidences = [int(conf) for conf in ocr_data["conf"] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        if avg_confidence > best_confidence and digits:
                            best_confidence = avg_confidence
                            best_raw_text = raw_text
                            best_sequence = digits  # Preserve leading zeros
                except Exception as e:
                    logger.warning(f"Tesseract PSM {psm_mode} failed: {e}")
                    continue
            
            if best_sequence and 5 <= len(best_sequence) <= 10:
                return best_sequence, best_confidence, best_raw_text
            return None, 0.0, best_raw_text
            
        except Exception as e:
            logger.error(f"Tesseract recognition failed: {e}")
            return None, 0.0, None

    def recognize_from_bytes(
        self, image_bytes: bytes, min_confidence: int = 60
    ) -> Tuple[Optional[str], float, Optional[str], str]:
        """
        Recognize digits from image bytes (legacy method, kept for compatibility).
        
        Args:
            image_bytes: Image file bytes
            min_confidence: Minimum confidence score threshold (0-100)

        Returns:
            Tuple of (reading_value_str, confidence_score, raw_text, status)
        """
        try:
            import tempfile

            logger.info("Starting OCR recognition from image bytes")

            # Save bytes to temp file and use regular recognize_digits
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = Path(tmp_file.name)
            
            try:
                result = self.recognize_digits(tmp_path, min_confidence=min_confidence)
                return result
            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except:
                    pass

        except Exception as e:
            logger.error(f"OCR processing from bytes failed: {str(e)}", exc_info=True)
            return None, 0.0, None, "error"

