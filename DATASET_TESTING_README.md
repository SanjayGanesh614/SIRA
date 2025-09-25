# Dataset Testing Feature

This document describes the new dataset testing functionality added to the SIRA Ocean Anomaly Detection System.

## Overview

Users can now upload their own oceanographic datasets and test them against the trained LSTM Autoencoder and XGBoost models for anomaly detection and event classification.

## Features Added

### 1. API Endpoints

#### Upload Dataset
- **Endpoint**: `POST /user/upload/dataset`
- **Description**: Upload a CSV dataset for model testing
- **Parameters**: 
  - `file`: CSV file upload
- **Returns**: Dataset path and filename

#### Test Model with Dataset
- **Endpoint**: `POST /user/test/model`
- **Description**: Test the anomaly detection model with uploaded dataset
- **Parameters**:
  - `dataset_filename`: Name of the uploaded dataset
- **Returns**: Analysis results and anomaly detection report

### 2. Dataset Validation

The system validates uploaded datasets to ensure they meet the requirements:

#### Required Columns
- `time`: Timestamp (any format)
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `depth`: Depth in meters
- `Temperature`: Water temperature
- `CTD_Salinity`: Salinity measurement
- `Oxygen_1`: Dissolved oxygen
- `CO2`: Carbon dioxide
- `TOC`: Total organic carbon
- `POC`: Particulate organic carbon
- `NO3_plus_NO2`: Nitrate + Nitrite
- `PO4`: Phosphate
- `Silicate`: Silicate concentration

#### Validation Rules
- Minimum 10 rows of data required
- Maximum 50% non-numeric values allowed in numeric columns
- Missing values will be filled with median values
- Non-numeric values will be converted where possible

### 3. User Interface

#### Streamlit Pages
1. **Models.py** - Updated with dataset testing tab
2. **dataset_testing.py** - Dedicated dataset testing page

#### Features
- File upload with drag-and-drop interface
- Data preview before upload
- Column validation with visual feedback
- Real-time model testing
- Detailed analysis results display
- Session management for multiple users

### 4. Model Integration

#### LSTM Autoencoder
- Anomaly detection using reconstruction error
- Depth-based thresholding
- Sequence-based analysis with 10-timestep windows

#### XGBoost Classifier
- Event type classification
- Marine heatwaves, hypoxia, nutrient blooms, etc.
- Confidence scoring

## Usage Instructions

### 1. Start the API Server
```bash
python main.py
```

### 2. Start the Streamlit Interface
```bash
streamlit run pages/Models.py
```

### 3. Upload and Test Dataset
1. Navigate to the "Dataset Testing" tab
2. Upload a CSV file with oceanographic data
3. Review the data preview and column validation
4. Click "Upload Dataset" to save the file
5. Click "Test Model" to run anomaly detection
6. View the detailed analysis results

## Example Dataset

An example dataset (`example_dataset.csv`) is provided with:
- 50 rows of synthetic oceanographic data
- All required columns present
- Realistic temperature, oxygen, and nutrient profiles
- Depth stratification from surface to 200m

## Technical Details

### File Structure
```
SIRA/
├── main.py                          # Updated with new endpoints
├── feature_detect.py                # Enhanced with validation functions
├── pages/
│   ├── Models.py                    # Updated with dataset testing
│   └── dataset_testing.py          # New dedicated page
├── example_dataset.csv             # Example dataset for testing
└── DATASET_TESTING_README.md       # This documentation
```

### Dependencies
- FastAPI for API endpoints
- Streamlit for user interface
- Pandas for data processing
- PyTorch for LSTM model
- XGBoost for classification
- Scikit-learn for preprocessing

## Error Handling

The system includes comprehensive error handling for:
- Invalid file formats
- Missing required columns
- Insufficient data
- Model loading failures
- API connection issues

## Future Enhancements

Potential improvements for future versions:
- Support for additional file formats (NetCDF, Excel)
- Batch processing of multiple datasets
- Custom model training with user data
- Advanced visualization of results
- Export functionality for analysis results

## Troubleshooting

### Common Issues
1. **"Dataset validation failed"**: Check that all required columns are present
2. **"Upload failed"**: Ensure the file is a valid CSV format
3. **"Testing failed"**: Verify the API server is running and accessible
4. **"Session not found"**: Refresh the page to create a new session

### Support
For technical support or questions about the dataset testing feature, please refer to the main project documentation or contact the development team.
