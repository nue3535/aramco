# aramco

## Objective
This product aims to develop an AI-driven diabetes management system to provide diabete predictions. By leveraging AI, this product will improve patient health outcomes, reduce the burden on healthcare providers, and enhance the accuracy of diabetes management decisions.

## ClearML Pipeline Execution Steps - Using Google Colab
For the end to end execution of this pipeline using google colab we would need 2 notebooks. First notebook is to work as agent for ClearML to execute the pipeline and in the second notebook we will execute the steps to actually run the pipeline after registering the task templates in ClearML.

### Notebook - 1
**Step 1 - Install clearml and clearml agent**
```
!pip install clearml
!pip install clearml-agent
```

**Step 2 - Initialize clearml**
```
!clearml-init
```

>**Note -** After running the above command you will be asked to pass your ClearML credentials in the terminal.

**Step 3 - Initialize clearml agent**
```
!clearml-agent init
```

**Step 4 - Run clearml agent daemon in background mode**
```
!clearml-agent daemon --queue "default" --detached
```
>**Note -** Change the queue name in the command above if you are using some other queue. For now using **default queue**



### Notebook - 2
**Step 1 - Clone the repository**
```
!git clone -b mlops-level1  https://github.com/nue3535/aramco.git
```

>**Note -** For now we are clonning branch **mlops-level1**. If you want to clone main branch use command
>
>```!git clone https://github.com/nue3535/aramco.git```

**Step 2 - Change your current directory to aramco**
```
%cd aramco/
```

**Step 3 - Install clearml**
```
!pip install clearml
```

**Step 4 - Initialize clearml**
```
!clearml-init
```
>**Note -** After running the above command you will be asked to pass your ClearML credentials in the terminal.

**Step 5 - Pre-register task templates in clearml one by one**
> **Note -** Make sure you comment out **"task.execute_remotely()"** in all the 6 scripts we are going to execute in this step
![image](https://github.com/user-attachments/assets/669d2ab3-7ce6-416b-9788-3b8364600816)

```
!python src/data_ingestion.py
```

```
!python src/data_preprocessing.py
```

```
!python src/hpo_tuning.py
```

```
!python src/model_training.py
```

```
!python src/model_evaluation.py
```

```
!python src/model_selection.py
```

After running all 6 scripts you should be able to see your registered tasks in ClearML dashboard like this
![image](https://github.com/user-attachments/assets/3cade609-0c53-462c-ae0b-665ebf6099c4)

> **Note -** After successfully registering all the tasks make sure to remove the comment **"task.execute_remotely()"** from all 6 scripts which we added before executing them.

**Step 6** - Executing the ClearML Pipeline
```
!python pipeline.py
```
After successful execution of your pipeline in ClearML you should be able to see it in your ClearML dashboard.
![image](https://github.com/user-attachments/assets/523e16ce-b3ae-47dd-af14-788a605ac3c2)


## Continuous Integration with GitHub Actions
We use GitHub Actions to automate our machine learning workflow. The CI pipeline performs the following tasks:

* Checks out the latest code on push to main
* Sets up the Python environment
* Installs all dependencies from requirements.txt
* Executes the ClearML training pipeline via pipeline.py
* Tracks experiment artifacts and performance through ClearML

![image](https://github.com/user-attachments/assets/69ba58bd-3329-4a42-9d64-c9a5d82d8caf)

## Streamlit User Interface (GUI)
Using Streamlit, we created a straightforward, interactive web interface that lets users enter medical information and get a diabetes prediction.

### Input Features
* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

### Output
* Predicted class: **Diabetic / Non-Diabetic**
![Screenshot 2025-05-23 172920](https://github.com/user-attachments/assets/6e948bcf-d498-4ee6-8771-6c05ae757d64)


## Local Deployment Instructions
Due to limited cloud resources, the application is deployed manually and run locally using Streamlit.
### How to Run the App Locally:
1. Ensure **rf_model.joblib** (best performing model in our case) is placed in the root directory or correct path
2. Install Streamlit
```
pip install streamlit
```

3. Launch the Streamlit app
```
streamlit run app.py
```

4. Open your browser and go to
```
http://localhost:8501
```

This allows real-time predictions.
