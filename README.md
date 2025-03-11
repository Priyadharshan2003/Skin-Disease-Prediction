# Skin Disease Prediction Project Deployment Guide

This guide explains how to deploy the Skin Disease Prediction application with the complete UI dashboard. The project uses advanced machine learning techniques to predict skin diseases from images and provides comprehensive analysis, home remedies, and nearby dermatologist locations.

## System Requirements

- Python 3.8+
- TensorFlow 2.5+
- Streamlit 1.10+
- Internet connection for geolocation services
- 8GB+ RAM (recommended for model inference)
- 4GB+ disk space

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skin-disease-prediction.git
   cd skin-disease-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained model**
   ```bash
   python download_model.py
   ```

## Dataset Setup

The project expects a dataset in the following structure:
```
smol/
├── train/
│   ├── Actinic keratosis/
│   ├── Atopic Dermatitis/
│   ├── Benign keratosis/
│   ├── Dermatofibroma/
│   ├── Melanocytic nevus/
│   ├── Melanoma/
│   ├── Squamous cell carcinoma/
│   ├── Tinea Ringworm/
│   ├── Candidiasis/
│   └── Vascular lesion/
└── val/
    ├── Actinic keratosis/
    ├── Atopic Dermatitis/
    ├── ...
```

Each directory should contain images of the respective skin condition.

## Training the Model (Optional)

If you want to train the model from scratch:

```bash
python train_model.py --epochs 20 --batch_size 32 --image_size 224
```

This will train the EfficientNetB0 model on your dataset and save the best model to `best_model.h5`.

## Running the Application

To start the Streamlit dashboard:

```bash
streamlit run app.py
```

This will launch the web interface on `http://localhost:8501`.

## Features Overview

1. **Skin Disease Prediction**
   - Upload skin images for analysis
   - Get predictions with confidence scores
   - View detailed information about the condition

2. **Analysis Dashboard**
   - Examine condition characteristics
   - View differential diagnosis
   - Explore similar cases statistics

3. **Home Remedies**
   - Access condition-specific home care suggestions
   - Important medical cautions and warnings
   - General skin care tips

4. **Nearby Dermatologists**
   - Find dermatologists within a specified radius
   - View on interactive map
   - Get contact information and ratings

## API Integration

To integrate with geolocation services for finding nearby dermatologists, you'll need to:

1. Register for an API key with a geolocation service provider (e.g., Google Maps, Mapbox)
2. Update the `config.py` file with your API key:
   ```python
   GEOLOCATION_API_KEY = "your_api_key_here"
   ```

## Security Considerations

- Ensure user data privacy by not storing uploaded images
- Implement proper authentication if deploying publicly
- Include clear disclaimers about medical advice

## Troubleshooting

- **Model loading errors**: Ensure the model file path is correct in `app.py`
- **Memory issues**: Reduce batch size or image dimensions
- **Geolocation errors**: Verify API key and internet connection

## Custom Model Integration

To use your own custom-trained model:

1. Train your model using the provided training script or your own methodology
2. Save the model in a compatible format (H5 or SavedModel)
3. Update the model path in the configuration

## Responsible Use Guidelines

- Always present this tool as a supplementary aid, not a replacement for professional medical diagnosis
- Include clear disclaimers about the accuracy and limitations
- Encourage users to seek professional medical advice for confirmed diagnoses

## License

This project is licensed under the MIT License - see the LICENSE file for details.
