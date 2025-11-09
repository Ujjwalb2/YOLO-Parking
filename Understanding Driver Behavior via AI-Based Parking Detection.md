This project provides an end-to-end pipeline for analyzing parking lot occupancy using computer vision and machine learning. It leverages a pre-trained YOLOv8 model for vehicle detection and the DBSCAN clustering algorithm to dynamically identify individual parking spots without manual annotation.

The system processes a dataset of parking lot images, detects all vehicles, groups these detections into logical parking spots, and then calculates occupancy statistics for each spot. Finally, it demonstrates a baseline Logistic Regression model to predict the next-frame occupancy of a spot based on time and its previous state.

🌟 Key Features
Vehicle Detection: Employs the powerful and efficient YOLOv8n model to detect vehicles (cars, motorcycles, buses, trucks).

Dynamic Spot Clustering: Uses DBSCAN on the center-points of detected vehicles to automatically identify and cluster parking spots, avoiding the need for manual coordinate mapping.

Occupancy Analytics: Aggregates detections per-spot to calculate key metrics like total_hits, unique_images, mean_confidence, and a normalized occupancy_rate.

Baseline Prediction: Implements a simple machine learning model (Logistic Regression) to predict spot occupancy, achieving a baseline AUC score.

Visualization: Generates bar charts for top-occupied spots and provides utility functions to draw detected spots on sample images.

⚙️ Project Workflow
The notebook is structured in the following sequential steps:

Setup and Data Ingestion

Installs all required Python libraries, including ultralytics, scikit-learn, pandas, and kagglehub.

Downloads the PKLot dataset from Kaggle using the kagglehub API.

Vehicle Detection (YOLOv8)

Loads the pre-trained yolov8n.pt model.

Runs batched inference on all images in the dataset to detect vehicles. The model is configured to detect classes 1 (motorcycle), 2 (car), 5 (bus), and 7 (truck).

Saves all raw detections (bounding boxes, confidence, class) to artifacts/predictions/predictions_flat.csv.

Parking Spot Identification (DBSCAN)

Reads the saved detections and normalizes the center coordinates (cx, cy) of all bounding boxes to a 0-1 scale.

Applies the DBSCAN clustering algorithm to these normalized coordinates. Each resulting cluster is considered a unique parking spot (spot_id).

Occupancy Analytics

Groups the clustered detections by the new spot_id.

Calculates aggregate statistics for each spot, including total detections, unique images the spot appeared in, and mean confidence.

Saves the final spot analytics to artifacts/analytics/spot_preference_vehicle_only.csv.

Visualization

Generates and saves bar charts (as .png files) visualizing the top 20 most-used spots, ranked by both total hits and occupancy rate.

Includes helper functions to load a random image and draw its detected bounding boxes, labeled with their assigned spot_id, to visually verify the clustering results.

Baseline Occupancy Prediction (Machine Learning)

As a final step, the notebook demonstrates a simple time-series prediction model.

It filters the data for a specific parking lot (UFPR04) and weather condition (sunny).

It creates features: hour (from the timestamp) and lag_occ (the spot's occupancy state in the previous frame).

It performs a time-based train/test split (using the last 20% of data for testing).

A LogisticRegression classifier is trained to predict the current occupancy (is_occ) based on the hour and lag_occ.

The model's performance is evaluated using the Area Under the Curve (AUC), providing a baseline for future occupancy prediction tasks.

🚀 How to Run
Prerequisites:

Python 3.x

A Kaggle account and API key (kaggle.json). The notebook uses kagglehub, which requires API credentials to be set up in your environment.

Installation: Clone this repository and install the required dependencies:

Bash

pip install -r requirements.txt
Execution:

Ensure your kaggle.json file is in the correct location (e.g., ~/.kaggle/kaggle.json).

Run the Jupyter Notebook Copy_of_Parking_Space_YOLO_NAS_Train_&_Predict (3).ipynb from top to bottom.

📂 Output Files
This project generates the following files and directories:

artifacts/

predictions/predictions_flat.csv: Raw CSV output of all vehicle detections from YOLOv8.

analytics/vehicle_predictions_with_spot_id.csv: The raw detections mapped to their corresponding clustered spot_id.

analytics/spot_preference_vehicle_only.csv: The final aggregated occupancy statistics for each unique spot_id.

figures/

top_spots_total_hits.png: Bar chart of the top 20 spots by total detection count.

top_spots_occupancy_rate.png: Bar chart of the top 20 spots by normalized occupancy rate.