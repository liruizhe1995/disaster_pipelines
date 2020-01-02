# Disaster Response Pipeline Project

### Project Description
In this project, I'll analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. I'll apply basic ETL and Machine Learning pipeline for this project. My project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### File Description
1. /app/run.py: the python code for running the web app
2. /data/disaster_message.csv: the csv file that contains original disaster messages.
3. /data/disaster_categories.csv: the csv file that contains original categories of disaster messages.
4. /data/process_data.py: Contains the ETL pipeline for processing disaster_message.csv and disaster_categories.csv and save the results into SQLite database called CleanedData.db.
5. /models/train_classifier.py: Contains machine learning pipeline for classying the database. The optimized classifier model will be saved called calssifier.pkl.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
