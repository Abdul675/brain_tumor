
# 🧠 Brain Tumor Detection using ResNet50 & Streamlit

This project uses **Transfer Learning with ResNet50** for automated **Brain Tumor Detection** from MRI scans.  
The trained model is integrated into a **Streamlit web application** that allows users to upload MRI images and view predictions with **Grad-CAM heatmaps**.

---

## 🚀 Features
- Deep Learning model using **ResNet50** (transfer learning on MRI dataset)  
- **Streamlit** web interface for real-time interaction  
- Upload MRI images and classify into:  
  - Glioma Tumor  
  - Meningioma Tumor  
  - Pituitary Tumor  
  - No Tumor  
- Grad-CAM visualization for model interpretability  
- Docker support for containerized deployment  
- Deployable on **Render / Docker Hub / Local machine**

---

## 📂 Project Structure
```

cnn_classifier/
│── main.py                # Streamlit app entry point
│── model.h5               # Pre-trained ResNet50 model
│── requirements.txt       # Python dependencies
│── Dockerfile             # Docker configuration
│── utils/                 # Helper functions (preprocessing, Grad-CAM, etc.)
│── static/                # Example images or assets
│── README.md              # Project documentation

````

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
````

### 2️⃣ Install Dependencies

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run main.py
```

Then open your browser at 👉 [http://localhost:8502](http://localhost:8502)

---

## 🐳 Run with Docker

### Build Docker Image

```bash
docker build -t brain_tumor_app .
```

### Run Container

```bash
docker run -d -p 8502:8502 --name brain_tumor_container brain_tumor_app
```

Now visit 👉 [http://localhost:8502](http://localhost:8502)

---

## 📊 Model Details

* **Architecture**: ResNet50 (Transfer Learning, fine-tuned layers)
* **Frameworks**: TensorFlow / Keras
* **Dataset**: Brain MRI scans (Tumor vs. Non-Tumor, 4 classes)
* **Training Environment**: Google Colab (weights exported and used in Streamlit app)

### Classification Report

```
                   precision    recall  f1-score   support

    glioma_tumor       0.89      0.86      0.88       165
meningioma_tumor       0.79      0.77      0.78       164
        no_tumor       0.90      0.96      0.93        79
 pituitary_tumor       0.86      0.88      0.87       165

        accuracy                           0.86       573
       macro avg       0.86      0.87      0.86       573
    weighted avg       0.85      0.86      0.85       573
```

---

## 🖼️ Grad-CAM Visualization

The model provides **visual explanations** using Grad-CAM, highlighting regions that influenced predictions.

Example:

| MRI Scan                      | Grad-CAM Heatmap               |
| ----------------------------- | ------------------------------ |
| ![MRI](static/sample_mri.jpg) | ![GradCAM](static/gradcam.jpg) |

---

## 🌍 Deployment

* **Localhost (default)** with Streamlit
* **Docker** for containerized deployment
* **Render / Docker Hub** for cloud deployment

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss what you’d like to change.

---

## 📜 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Abdul Hameed**
Data Science & AI Enthusiast

🔗 [LinkedIn](https://www.linkedin.com/in/abdul-hameed-6119561b0/) | [GitHub](https://github.com)

```
```
