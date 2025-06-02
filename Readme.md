# Water Pump Functional Status Predictor

This project predicts the **functional status** of water pumps in Tanzania using machine learning. It compares three different classifiers and serves predictions through a user-friendly **Streamlit interface** with a **FastAPI backend**.

---

## 🚀 Overview

- **Frontend**: Built with [Streamlit](https://streamlit.io/)
- **Backend**: Powered by [FastAPI](https://fastapi.tiangolo.com/)
- **ML Models**: Decision Tree, Random Forest, XGBoost
- **Preprocessing**: Column-wise pipeline for data transformation
- **Deployment**: Dockerized using Docker Desktop and Docker Hub

---

## 🧠 Model Comparison

| Model         | Train Accuracy | Test Accuracy |
|---------------|----------------|---------------|
| Decision Tree | 0.7672         | 0.7496        |
| Random Forest | 0.8141         | 0.7765        |
| XGBoost       | 0.8607         | 0.8068        |

XGBoost performed best overall.

---

## 📁 Project Structure (Important files)

| File                  | Purpose                                       |
|-----------------------|-----------------------------------------------|
| `code.ipynb`          | code file to train the model                  |
| `*_.csv`              | data files for training/testing               |
| `app.py`              | Streamlit app to capture user input           |
| `api_app.py`          | FastAPI backend to handle predictions         |
| `helper_function.py`  | Functions used in the prediction pipeline     |
| `preprocessor.joblib` | ColumnTransformer preprocessing pipeline      |
| `*_model.joblib`      | Trained ML models (DecisionTree, RF, XGBoost) |
| `requirements.txt`    | Python dependencies                           |
| `docker-compose.yml`  | Docker config to run the full stack           |
| `categories.json`     | categories of each column used for training   |
| `logger_seup.py`      | file and console logging| steup               |
| `Dockerfile.api`      | instructions for api docker image             |
| `Dockerfile.streamlit`| instructions for streamlit docker image       |
| `docker-compose.yml`  | file for multi-container Docker applications  |

---

## 🛠️ How It Works

1. **User Input (Frontend)**: Users enter pump details in Streamlit.
2. **Data Transfer**: Streamlit sends this input to the FastAPI backend.
3. **Prediction (Backend)**: The backend applies preprocessing and runs the input through a trained model.
4. **Output**: The result (functional, needs repair, non-functional) is sent back and displayed in the Streamlit UI.

---

## ⚙️ Setup Instructions

You have **two options** to run this project:

---

### 🔧 Option 1: Manual Setup Using `requirements.txt`

> Recommended for development or when Docker is not available.

#### ✅ Steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/shakeeluetp1041/DataDrip.git
   cd water-pump-predictor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI backend:
   ```bash
   uvicorn api_app:app --reload
   ```

5. Open a new terminal and run Streamlit:
   ```bash
   streamlit run app.py
   ```

6. Navigate to:
   - FastAPI Docs: [http://localhost:8000](http://localhost:8000)
   - Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

### 🐳 Option 2: Run Using Docker

> Recommended for clean, platform-independent setup.

#### ✅ Prerequisites:
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/macOS/Linux)

#### ✅ Steps:

1. Open **VS Code** (or any IDE) and create a new folder.

2. Inside the folder, create a file named `docker-compose.yml` with the following content:

    ```yaml
    services:
      streamlit:
        image: shakeelahmed1041/streamlit-datadrip:latest
        ports:
          - "8501:8501"
        depends_on:
          - fastapi
        restart: unless-stopped

      fastapi:
        image: shakeelahmed1041/fastapi-datadrip:latest
        ports:
          - "8000:8000"
        restart: unless-stopped
    ```

3. Open the terminal in that folder and run:

    ```bash
    docker pull shakeelahmed1041/streamlit-datadrip:latest
    docker pull shakeelahmed1041/fastapi-datadrip:latest
    docker-compose up
    ```

4. In your browser:
   - Open FastAPI: [http://localhost:8000](http://localhost:8000)
   - Open Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## ✅ Requirements

To run this project (manually or via Docker), you’ll need:

- Python 3.11+
- pip (Python package manager)
- Docker Desktop (optional, for Docker method)
- Internet connection (to pull Docker images or install packages)

---

## 📬 Feedback

Feel free to open issues or pull requests if you'd like to contribute or improve this project!
