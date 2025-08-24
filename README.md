# Photo Labeler

A privacy-focused, local-only photo labeling and organization tool that automatically generates descriptive filenames based on image content using computer vision models for image captioning and classification. All processing happens on your machine - your photos never leave your computer.

## Features

- **Computer Vision Analysis**: Uses BLIP (Bootstrapping Language-Image Pre-training) for caption generation and Vision Transformer for image classification
- **Smart Filename Generation**: Creates descriptive filenames by filtering out common stop words and focusing on meaningful descriptive content
- **Color Analysis**: Extracts dominant colors using K-means clustering with proper color science (HSV color space)
- **EXIF Metadata Extraction**: Safely extracts image metadata with privacy-conscious filtering of sensitive data
- **Batch Processing**: Process entire folders of images efficiently
- **File Renaming**: Optionally rename files with generated labels
- **Fast Mode**: Skip computer vision analysis for quicker metadata-only processing
- **Configurable**: Adjustable parameters for file size limits, confidence thresholds, and filename length

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd photo-labeler
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch transformers opencv-python pillow scikit-learn
```

## Usage

### Basic Usage

Process all images in a folder and generate analysis results:
```bash
python label.py /path/to/your/photos
```

### Advanced Usage

```bash
# Rename files based on generated labels
python label.py /path/to/photos --rename

# Fast mode (metadata and colors only, no computer vision analysis)
python label.py /path/to/photos --fast

# Custom output file
python label.py /path/to/photos -o my_results.json

# Adjust parameters
python label.py /path/to/photos --max-size 100 --confidence 0.2 --max-filename-length 60
```

### Configuration Options

- `--rename` or `-r`: Rename files based on generated labels
- `--fast`: Skip computer vision analysis for faster processing
- `--output` or `-o`: Specify custom output JSON file
- `--max-size`: Maximum image size in MB (default: 50)
- `--confidence`: Minimum confidence threshold for tags (default: 0.1)
- `--max-filename-length`: Maximum generated filename length (default: 50)

## Examples

### Before and After Filename Improvement

**Original truncated names:**
- `the_building_is_085.jpg` (cut off after "is")
- `a_view_of the city_082.jpg` (incomplete description)
- `a_bathroom_with_069.jpg` (cut off after "with")

**Improved descriptive names:**
- `building_pink_palace.jpg` (complete, descriptive)
- `view_city_top_building.jpg` (complete, descriptive)  
- `bathroom_toilet_and_drawing.jpg` (complete, descriptive)

### Sample Output

```json
{
  "analysis_date": "2025-01-15T10:30:00",
  "folder_path": "/Users/example/Photos",
  "total_images": 50,
  "results": [
    {
      "original_path": "/Users/example/Photos/IMG_001.jpg",
      "filename": "IMG_001.jpg",
      "caption": "a group of children playing on a water slide",
      "tags": ["playground", "water"],
      "colors": ["cyan", "yellow", "blue"],
      "suggested_label": "group_children_playing_water",
      "metadata": {
        "DateTime": "2025:01:10 14:30:00",
        "Make": "Apple",
        "Model": "iPhone 15"
      }
    }
  ]
}
```

## Project Structure

```
photo-labeler/
├── label.py           # Main photo labeling script
├── README.md          # This file
├── LICENSE           # MIT license
└── venv/             # Virtual environment (created during setup)
```

## Dependencies

### Core Libraries
- **OpenCV** (`opencv-python`): Image processing and color analysis
- **Pillow** (`pillow`): Image loading and EXIF data extraction
- **NumPy**: Numerical operations (installed with OpenCV)

### Machine Learning Libraries (Optional for fast mode)
- **PyTorch** (`torch`): Deep learning framework
- **Transformers** (`transformers`): Hugging Face transformers for ML models

### Optional Libraries
- **scikit-learn**: K-means clustering for color analysis
- **argparse**: Command-line interface (built-in)
- **pathlib**: Path handling (built-in)

## Privacy & Security

### Your Photos Never Leave Your Machine
This tool is designed with privacy as a core principle - **all photo analysis happens 100% locally on your computer**. Your images are never uploaded to any external service or cloud API. This ensures:
- Complete privacy of your personal photos
- No dependency on internet connectivity after initial setup
- No usage limits or API costs
- Full control over your data

### Internet Usage (First Run Only)
The application requires internet access **only once** on first run to download the ML models from Hugging Face's model hub:
- Models are downloaded and cached locally in `~/.cache/huggingface/`
- After initial download, the tool works completely offline
- Use `--fast` mode to skip computer vision analysis entirely (no models needed)
- Downloaded models are about 500MB total

### Security Features
- All processing happens locally - photos never leave your device
- Sensitive EXIF data (GPS, user comments, etc.) filtered by default
- Path validation prevents directory traversal attacks
- File size limits prevent memory exhaustion

## Technical Details

### Machine Learning Models Used
- **Image Captioning**: Salesforce BLIP (`Salesforce/blip-image-captioning-base`)
- **Image Classification**: Google Vision Transformer (`google/vit-base-patch16-224`)

### Smart Filename Generation
The tool uses an intelligent filename generation algorithm that:
1. Filters out common stop words (`a`, `the`, `is`, `of`, etc.)
2. Prioritizes meaningful descriptive words
3. Uses smart truncation that preserves complete descriptions
4. Falls back to tags and colors when captions aren't available

### Privacy and Security
- Filters sensitive EXIF data (GPS, user comments, etc.) by default
- Validates file paths to prevent directory traversal attacks
- Limits file sizes to prevent out-of-memory errors

## Performance Notes

- **Memory Usage**: ML models require ~2-4GB RAM
- **Processing Speed**: ~2-5 seconds per image with computer vision analysis, <1 second in fast mode
- **GPU Support**: Automatically uses MPS (Mac) or CUDA if available

## Troubleshooting

### Common Issues

**"Missing required core library" error:**
```bash
pip install opencv-python pillow
```

**"ML libraries not available" warning:**
```bash
pip install torch transformers
```

**Out of memory errors:**
- Reduce `--max-size` parameter
- Use `--fast` mode for large batches
- Process smaller batches of images

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) for image captioning
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model integration
- [Google Vision Transformer](https://huggingface.co/google/vit-base-patch16-224) for image classification