# 🦠 Malaria Cell Detection App

A deep learning web application built with **Streamlit** that classifies microscopic cell images as **Infected** or **Uninfected** using multiple models, with optional **Grad-CAM visualization** for interpretability.

---

## 🚀 Live Demo

👉 *(Add your Streamlit link here after deployment)*
`https://your-app-name.streamlit.app`

---

## 📌 Features

* 🧠 **Multiple Model Comparison**

  * Custom CNN
  * MobileNetV2 (Frozen)
  * MobileNetV2 (Fine-tuned)

* 📊 **Confidence Scores**

  * Displays prediction confidence for each model

* 🔍 **Grad-CAM Visualization**

  * Highlights regions of the image most important for model decisions

* 🖼️ **Image Upload Support**

  * Upload `.jpg`, `.png`, or `.jpeg` cell images

---

## 🧪 How It Works

1. Upload a microscopic cell image
2. The app preprocesses the image
3. Each model makes a prediction
4. Results are displayed side-by-side
5. (Optional) Grad-CAM shows what the model focused on

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV
* **Visualization:** Grad-CAM
* **Model Hosting:** Google Drive (via gdown)

---

## 📂 Project Structure

```
malaria-streamlit-app/
│
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
├── assets/
│   ├── uninfected_example.png
│   └── parasitized_example.png
└── README.md           # Project documentation
```

---

## ⚙️ Installation (Run Locally)

### 1. Clone the repository

```
git clone https://github.com/yourusername/malaria-streamlit-app.git
cd malaria-streamlit-app
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app
> **Windows / Mac users**:
``` bash
streamlit run app.py
```
> **Linux / GitHub Codespaces / WSL users**: you need OpenGL for image rendering
```bash
sudo apt-get update
sudo apt-get install -y libgl1
python3 -m streamlit run app.py
```

---

## ⏳ Notes

* Model files are automatically downloaded on first run using `gdown`
* Initial load may take **30–60 seconds**

---

## 🔗 Related Project

Full model training, experimentation, and notebooks:
👉 `https://github.com/MaryvilleUniversity-AI/malaria-image-classification`

---

## 📸 Screenshots

*(Add screenshots to an /assets folder and link them here)*

```
![App Demo](assets/demo.png)
![Grad-CAM Example](assets/gradcam_example.png)
```

---

## 💡 Future Improvements

* Add more model architectures
* Improve UI/UX design
* Add batch image processing
* Deploy with Docker

---

## 👤 Author

MaryvilleUniversity-AI
GitHub: [https://github.com/yourusername](https://github.com/MaryvilleUniversity-AI)

---

## ⭐ Acknowledgments

* Dataset: NIH Malaria Dataset
* Inspiration: Medical imaging and AI interpretability research

---

## 📜 License

This project is open source and available under the MIT License.
