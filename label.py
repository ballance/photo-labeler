import os
import json
import re
import colorsys
from pathlib import Path
import argparse
from datetime import datetime
from collections import Counter

# Core dependencies
try:
    import cv2
    import numpy as np
    from PIL import Image, ExifTags
except ImportError as e:
    print(f"Missing required core library: {e}")
    print("Install with: pip install opencv-python pillow")
    exit(1)

HAS_AI_LIBS = True
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import pipeline
except ImportError:
    HAS_AI_LIBS = False

# Optional ML libraries
HAS_SKLEARN = True
try:
    from sklearn.cluster import KMeans
except ImportError:
    HAS_SKLEARN = False

# Default configuration
DEFAULT_CONFIG = {
    'max_image_size': 50 * 1024 * 1024,  # 50MB
    'caption_max_length': 50,
    'top_k_tags': 5,
    'confidence_threshold': 0.1,
    'filename_max_length': 50,
    'filter_sensitive_exif': True,
    'color_clusters': 5
}

class PhotoLabeler:
    def __init__(self, fast_mode=False, config=None):
        """Initialize the photo labeler.
        
        Args:
            fast_mode: If True, skip AI model loading for faster processing
            config: Configuration dictionary for customization
        """
        self.fast_mode = fast_mode
        self.config = config or {}
        
        self._caption_processor = None
        self._caption_model = None
        self._classifier = None
        
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(self.config)
        self.config = final_config
        
        self.max_image_size = self.config['max_image_size']
        self.caption_max_length = self.config['caption_max_length']
        self.top_k_tags = self.config['top_k_tags']
        self.confidence_threshold = self.config['confidence_threshold']
        self.filename_max_length = self.config['filename_max_length']
        self.filter_sensitive_exif = self.config['filter_sensitive_exif']
        self.color_clusters = self.config['color_clusters']
        
        if not fast_mode and not HAS_AI_LIBS:
            print("Warning: AI libraries not available. Running in fast mode.")
            print("Install with: pip install torch transformers")
            self.fast_mode = True
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        if hasattr(self, '_caption_model') and self._caption_model is not None:
            del self._caption_model
            self._caption_model = None
        
        if hasattr(self, '_caption_processor') and self._caption_processor is not None:
            del self._caption_processor
            self._caption_processor = None
            
        if hasattr(self, '_classifier') and self._classifier is not None:
            del self._classifier
            self._classifier = None
        
        # Clear torch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def _load_caption_models(self):
        """Lazy load caption generation models."""
        if self._caption_processor is None:
            print("Loading caption models...")
            self._caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    def _load_classifier(self):
        """Lazy load classification model."""
        if self._classifier is None:
            print("Loading classification model...")
            self._classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    
    def get_image_caption(self, image_path):
        """Generate a descriptive caption for the image."""
        if self.fast_mode:
            return None
            
        try:
            self._load_caption_models()
            image = Image.open(image_path).convert('RGB')
            inputs = self._caption_processor(image, return_tensors="pt")
            
            with torch.no_grad():
                out = self._caption_model.generate(**inputs, max_length=self.caption_max_length, num_beams=5)
            
            caption = self._caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return None
    
    def get_image_tags(self, image_path):
        """Get classification tags for the image."""
        if self.fast_mode:
            return []
            
        try:
            self._load_classifier()
            image = Image.open(image_path).convert('RGB')
            results = self._classifier(image, top_k=self.top_k_tags)
            
            # Extract labels with confidence > threshold
            tags = [result['label'] for result in results if result['score'] > self.confidence_threshold]
            return tags
        except Exception as e:
            print(f"Error getting tags for {image_path}: {e}")
            return []
    
    def _validate_image_size(self, image_path):
        """Validate image file size to prevent OOM."""
        try:
            file_size = os.path.getsize(image_path)
            if file_size > self.max_image_size:
                print(f"Warning: {os.path.basename(image_path)} is {file_size/1024/1024:.1f}MB, skipping (max: {self.max_image_size/1024/1024}MB)")
                return False
            return True
        except Exception:
            return False
    
    def _validate_path(self, path):
        """Validate file path for security."""
        try:
            # Resolve path and check it's within expected bounds
            resolved_path = Path(path).resolve()
            # Basic path traversal protection
            if '..' in str(path) or str(resolved_path).startswith('/etc') or str(resolved_path).startswith('/root'):
                print(f"Warning: Potentially unsafe path: {path}")
                return False
            return resolved_path.exists()
        except Exception:
            return False
    
    def extract_metadata(self, image_path, filter_sensitive=True):
        """Extract EXIF metadata from the image.
        
        Args:
            image_path: Path to image file
            filter_sensitive: If True, filter out potentially sensitive EXIF data
        """
        try:
            image = Image.open(image_path)
            exif_data = {}
            
            # Sensitive EXIF tags to filter out
            sensitive_tags = {'GPS', 'GPSInfo', 'UserComment', 'ImageDescription', 
                            'Artist', 'Copyright', 'Software'} if filter_sensitive else set()
            
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if tag not in sensitive_tags:
                            # Convert complex objects to strings for JSON serialization
                            try:
                                json.dumps(value)
                                exif_data[tag] = value
                            except (TypeError, ValueError):
                                exif_data[tag] = str(value)
            
            return exif_data
        except Exception as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return {}
    
    def _rgb_to_color_name(self, r, g, b):
        """Convert RGB values to color name using HSV color space."""
        # Normalize RGB to 0-1 range
        r, g, b = r/255.0, g/255.0, b/255.0
        
        # Convert to HSV
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        
        # Convert hue to degrees
        h = h * 360
        
        # Determine color based on HSV values
        if v < 0.2:  # Very dark
            return "black"
        elif v > 0.8 and s < 0.2:  # Very bright and low saturation
            return "white"
        elif s < 0.2:  # Low saturation (grayscale)
            if v < 0.4:
                return "dark_gray"
            elif v > 0.7:
                return "light_gray"
            else:
                return "gray"
        else:
            # Color classification by hue
            if h < 15 or h >= 345:
                return "red"
            elif h < 45:
                return "orange"
            elif h < 75:
                return "yellow"
            elif h < 165:
                return "green"
            elif h < 195:
                return "cyan"
            elif h < 255:
                return "blue"
            elif h < 285:
                return "purple"
            elif h < 315:
                return "magenta"
            else:
                return "pink"
    
    def analyze_colors(self, image_path):
        """Analyze dominant colors in the image using proper color science."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use k-means to find dominant colors if sklearn available
            if HAS_SKLEARN:
                n_clusters = min(self.color_clusters, len(pixels) // 100)  # Ensure enough pixels per cluster
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(pixels)
                    colors = kmeans.cluster_centers_.astype(int)
                else:
                    colors = [np.mean(pixels, axis=0).astype(int)]
            else:
                # Fallback: sample colors from image
                step = max(1, len(pixels) // 1000)  # Sample up to 1000 pixels
                colors = pixels[::step][:10]  # Take up to 10 sample colors
            
            # Convert RGB colors to color names using HSV
            color_names = []
            for color in colors:
                r, g, b = color
                color_name = self._rgb_to_color_name(r, g, b)
                color_names.append(color_name)
            
            # Return unique color names, most common first
            color_counts = Counter(color_names)
            return [color for color, count in color_counts.most_common()]
            
        except Exception as e:
            print(f"Error analyzing colors for {image_path}: {e}")
            return []
    
    def _sanitize_filename(self, text):
        """Sanitize text for use in filenames."""
        if not text:
            return ""
        
        # Remove or replace invalid filename characters
        text = re.sub(r'[<>:"/\\|?*]', '', text)  # Remove invalid chars
        text = re.sub(r'[^\w\s-]', '', text)  # Keep only alphanumeric, spaces, hyphens
        text = re.sub(r'\s+', '_', text)  # Replace spaces with underscores
        text = re.sub(r'_+', '_', text)  # Collapse multiple underscores
        text = text.strip('_')  # Remove leading/trailing underscores
        
        return text.lower()
    
    def generate_filename_label(self, analysis_result):
        """Generate a descriptive filename based on analysis with proper sanitization."""
        caption = analysis_result.get('caption', '')
        tags = analysis_result.get('tags', [])
        colors = analysis_result.get('colors', [])
        
        # Create a descriptive name with multiple fallbacks
        filename_parts = []
        
        # Try caption first
        if caption:
            # Take meaningful words from caption, prioritizing descriptive content
            words = caption.lower().split()
            # Filter out common articles and prepositions that aren't descriptive
            skip_words = {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'on', 'at', 'with', 'by', 'for', 'to', 'from'}
            meaningful_words = [w for w in words if w and w.isalpha() and w not in skip_words]
            # Take up to 4 meaningful words to get better descriptions
            clean_words = [self._sanitize_filename(w) for w in meaningful_words[:4] if w]
            if clean_words:
                filename_parts.extend(clean_words)
        
        # Add top tags if we have space
        if tags and len(filename_parts) < 3:
            clean_tags = [self._sanitize_filename(tag) for tag in tags[:2] if tag]
            filename_parts.extend(clean_tags)
        
        # Add dominant color if we still need more descriptors
        if colors and len(filename_parts) < 3:
            filename_parts.append(colors[0])
        
        # Join parts and enforce length limit
        if filename_parts:
            result = "_".join(filename_parts)
            # Truncate if too long, but prioritize keeping meaningful content
            if len(result) > self.filename_max_length:
                # Instead of cutting at word boundaries, try to keep the most descriptive parts
                # If we have a caption-based description, prioritize keeping it complete
                if len(filename_parts) > 1:
                    # Try combinations, starting with full caption words
                    for num_parts in range(len(filename_parts), 0, -1):
                        test_result = "_".join(filename_parts[:num_parts])
                        if len(test_result) <= self.filename_max_length:
                            result = test_result
                            break
                    else:
                        # If even single parts are too long, truncate the first part intelligently
                        result = filename_parts[0][:self.filename_max_length-4] + "_trunc"
                else:
                    # Single part too long - truncate but keep it readable
                    result = result[:self.filename_max_length-4] + "_trunc"
            return result
        
        return "unlabeled"
    
    def process_image(self, image_path):
        """Process a single image and return analysis results."""
        print(f"Processing: {os.path.basename(image_path)}")
        
        # Validate image first
        if not self._validate_path(image_path) or not self._validate_image_size(image_path):
            return None
        
        analysis = {
            'original_path': str(image_path),
            'filename': os.path.basename(image_path),
            'timestamp': datetime.now().isoformat(),
            'caption': self.get_image_caption(image_path),
            'tags': self.get_image_tags(image_path),
            'colors': self.analyze_colors(image_path),
            'metadata': self.extract_metadata(image_path, filter_sensitive=self.filter_sensitive_exif)
        }
        
        # Generate suggested label
        analysis['suggested_label'] = self.generate_filename_label(analysis)
        
        return analysis
    
    def process_folder(self, folder_path, output_file=None, rename_files=False):
        """Process all images in a folder."""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"Error: Folder '{folder_path}' does not exist.")
            return
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.glob(f"*{ext}"))
            image_files.extend(folder_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in '{folder_path}'")
            return
        
        print(f"Found {len(image_files)} image files to process.")
        
        results = []
        renamed_files = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] ", end="")
            
            try:
                analysis = self.process_image(image_path)
                if analysis is None:  # Skip if validation failed
                    continue
                    
                results.append(analysis)
                
                # Rename file if requested
                if rename_files and analysis['suggested_label'] != 'unlabeled':
                    old_path = Path(image_path)
                    extension = old_path.suffix
                    new_name = f"{analysis['suggested_label']}_{i:03d}{extension}"
                    new_path = old_path.parent / new_name
                    
                    # Avoid overwriting existing files
                    counter = 1
                    while new_path.exists():
                        new_name = f"{analysis['suggested_label']}_{i:03d}_{counter}{extension}"
                        new_path = old_path.parent / new_name
                        counter += 1
                    
                    try:
                        old_path.rename(new_path)
                        renamed_files.append({
                            'old_name': old_path.name,
                            'new_name': new_path.name,
                            'label': analysis['suggested_label']
                        })
                        print(f"Renamed to: {new_path.name}")
                    except Exception as e:
                        print(f"Error renaming file: {e}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save results to JSON file
        if output_file is None:
            output_file = folder_path / "photo_analysis_results.json"
        
        output_data = {
            'analysis_date': datetime.now().isoformat(),
            'folder_path': str(folder_path),
            'total_images': len(image_files),
            'results': results,
            'renamed_files': renamed_files if rename_files else []
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n\nAnalysis complete!")
        print(f"Results saved to: {output_file}")
        if rename_files:
            print(f"Renamed {len(renamed_files)} files")
        
        # Print summary
        self.print_summary(results)
    
    def print_summary(self, results):
        """Print a summary of the analysis results."""
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Most common tags
        all_tags = []
        for result in results:
            all_tags.extend(result.get('tags', []))
        
        if all_tags:
            tag_counts = Counter(all_tags)
            print(f"\nMost common tags:")
            for tag, count in tag_counts.most_common(10):
                print(f"  {tag}: {count}")
        
        # Sample captions
        captions = [r.get('caption') for r in results if r.get('caption')]
        if captions:
            print(f"\nSample captions:")
            for caption in captions[:5]:
                print(f"  • {caption}")

def main():
    parser = argparse.ArgumentParser(description="Automatically label photos based on content")
    parser.add_argument("folder", help="Path to folder containing photos")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument("-r", "--rename", action="store_true", 
                       help="Rename files based on generated labels")
    parser.add_argument("--fast", action="store_true",
                       help="Fast mode: skip AI analysis (metadata and colors only)")
    parser.add_argument("--max-size", type=int, default=50,
                       help="Maximum image size in MB (default: 50)")
    parser.add_argument("--confidence", type=float, default=0.1,
                       help="Minimum confidence threshold for tags (default: 0.1)")
    parser.add_argument("--max-filename-length", type=int, default=50,
                       help="Maximum generated filename length (default: 50)")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'max_image_size': args.max_size * 1024 * 1024,  # Convert MB to bytes
        'confidence_threshold': args.confidence,
        'filename_max_length': args.max_filename_length
    }
    
    # Initialize and run the labeler
    with PhotoLabeler(fast_mode=args.fast, config=config) as labeler:
        labeler.process_folder(args.folder, args.output, args.rename)

if __name__ == "__main__":
    main()
