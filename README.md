# Predicting Accident Severity using US Accidents Dataset

This project involves analyzing and processing the US Accidents dataset to predict accident severity using machine learning techniques. By leveraging historical accident data, we aim to develop models to assist in enhancing road safety.

## Dataset
We use the [US Accidents dataset](https://drive.google.com/file/d/1edKrdWNOcgbAo2JtckX-PEyM0FdEq4EG/view?usp=drive_link), which provides detailed information about road traffic accidents in the United States. The dataset includes attributes such as location, weather conditions, time of occurrence, and severity levels.

## Features
The dataset contains various features, including:

- **Numerical Features:**
  - Start_Lat, Start_Lng
  - End_Lat, End_Lng
  - Distance(mi), Temperature(F)
  - Wind_Chill(F), Humidity(%), and more

- **Categorical Features:**
  - Source, Weather_Condition
  - Sunrise_Sunset, Civil_Twilight, etc.

## Project Objectives
1. **Data Cleaning & Preprocessing**: Handle missing values and outliers in the dataset.
2. **Exploratory Data Analysis (EDA)**: Identify patterns and relationships in accident occurrences.
3. **Model Development**: Train machine learning models to predict accident severity.
4. **Model Evaluation**: Assess the model performance on accuracy and precision metrics.
5. **Recommendations**: Provide insights and actionable strategies to mitigate accidents.

---

## File Structure

```
.
├── preprocess_chunk: Function to clean each chunk of data.
├── download_dataset: Download dataset from Google Drive.
├── load_and_preprocess_data: Process the entire dataset in chunks for efficiency.
├── main: Primary execution flow.
```

---

## Usage
### Pre-requisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `gdown`

### Execution
1. Install the dependencies using:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn gdown
   ```

2. Run the script:
   ```bash
   python script_name.py
   ```

3. Review the output classification report and accuracy score.

---

## Key Functions

### `preprocess_chunk(chunk)`
- Fills missing values in numerical and categorical features.
- Extracts year information from the `Start_Time` column.
- Drops unused columns to streamline processing.

### `download_dataset(file_id, output_file)`
- Downloads the dataset from Google Drive using its unique ID.

### `load_and_preprocess_data(file_path, chunk_size=10000)`
- Loads the dataset in chunks for memory efficiency.
- Applies the `preprocess_chunk` function to each chunk.

### `main()`
- Downloads the dataset.
- Loads and preprocesses the data.
- Splits data into training and testing sets.
- Trains a Random Forest model and evaluates its performance.

---

## Results
### Evaluation Metrics:
- **Classification Report:** Precision, Recall, F1-Score for each severity class.
- **Accuracy Score:** Overall model accuracy.

---

## Recommendations
- **Feature Importance**: Further analysis of influential factors.
- **Additional Models**: Try other models such as XGBoost or Neural Networks for comparison.
- **Road Safety Strategies**: Insights from EDA can inform decision-making to reduce accidents.

---

## Acknowledgements
- Dataset: Provided by Kaggle [US Accidents](https://drive.google.com/file/d/1edKrdWNOcgbAo2JtckX-PEyM0FdEq4EG/view?usp=drive_link).

## Contributing
Feel free to fork this repository and suggest improvements!
