# Multi-Label News Classifier

## Overview
The **Multi-Label News Classifier** is a machine learning project designed to classify news articles into multiple categories. It processes datasets, trains models, and evaluates their performance to predict labels for news articles. The project supports multi-label classification, where a single article can belong to more than one category. Moved to Multi-Label News Classifier due to the wrong predicted title from refined model can be more than one label (Multi label) after thorough analysis.
## Features
- Preprocessing of datasets (CSV and JSON formats).
- Mapping and refining datasets for specific categories.
- Multi-label classification using machine learning models.
- Evaluation and refinement of model performance.
- Exporting results for further analysis.

### Key Files
- **`dataset/`**: Contains datasets used for training, fine-tuning, and testing.
- **`refine_model.py`**: Script for refining the model and saving test results.
- **`test_refined.py`**: Script for evaluating refined model (One Label).
- **`multi_label.py`**: Script for training and modify to multi label classifier.
- **`test_refine_model.py`**: Script for testing the multi label model.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

## Datasets
The project uses datasets from various sources, including:
- AG News dataset for topic classification.
- Kaggle: NYT Articles
- Newsdata.io
- Custom datasets for fine-tuning and testing.

## Workflow
1. **Data Preprocessing**:
   - Datasets are cleaned, mapped, and refined using scripts like `DataPreprocessCSV.py` and `nytPreprocess.py`.
   - Categories are mapped to labels for consistency.

2. **Model Training**:
   - Models are trained on balanced datasets to ensure fair representation of all categories.

3. **Evaluation**:
   - Test results are saved to files like `test_results_refined.csv` for analysis.

4. **Refinement**:
   - Models are fine-tuned using additional datasets to improve accuracy.

## Labels (Classes)
   The model predicts from the following 8 classes:

   - Arts, Culture, and Entertainment
   - Business and Finance
   - Health and Wellness
   - Lifestyle and Fashion
   - Politics
   - Science and Technology
   - Sports
   - Crime

## Results
The project outputs predictions in CSV format, such as:

- test_results_refined.csv: Contains predicted and actual labels for test data.

The multi-label model accepts text input and returns multiple predicted labels from the classes above.

## Model Files
**Note:**
Due to file size limitation (~500MB) the trained model files are stored externally on Google Dive.
You can download them from the link in saved_models README file and place them in the saved_models directory.

## Getting Started
1. Clone this repository.
2. Install dependencies:
   pip install -r requirements.txt
3. Donwload the saved model from Google Drive.
4. Run a prediction test:
   python test_refine_model.py

## Future Work
- Incorporate more datasets for better generalization.
- Experiment with advanced deep learning models for improved accuracy.
-  Deploy the model via a web interface or API

## License
This project is intended for educational and portfolio purposes. 

## 🚀 Thank you for visiting!
