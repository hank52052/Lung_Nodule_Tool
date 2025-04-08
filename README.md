# Lung Nodule Diagnosis Tool

This tool provides an AI-based workflow for lung nodule visualization and malignancy prediction using chest CT scans.

---

## 🔍 Description

Users can upload:

- A **3D CT image** file (`.nrrd`, `.nii`, or `.nii.gz`) of the patient’s lung
- A corresponding **contour file** (`.nrrd`, `.nii`, or `.nii.gz`) marking the suspected nodule region

Once uploaded, the system will:

- Automatically apply **lung window display**
- Show the **CT slice with overlaid contour**
- Allow **Z-axis scrolling** (slice navigation for 3D volume)
- Allow **contour toggle** (show/hide)
- Perform **malignancy prediction** (Benign vs Malignant)

---

## 📦 Input Format

- CT scan: `image.nrrd`
- Contour: `roi_1.nrrd` (or any mask with same dimensions)

---

## 🚀 Prediction Output

- Visual result with nodule location highlighted
- Confidence percentage of:
  - **Benign nodule**
  - **Malignant nodule**

---

## 💻 Compatibility

Runs directly in the browser on:
- Desktop
- Tablet
- Mobile

## 📦 Requirements

Before running the tool, make sure you have all the required Python packages installed.

```bash
pip install -r requirements.txt
```
🔎 You can find the requirements.txt file in this repository.

## 📁 Folder Structure
To run the program successfully, all project files must be placed in the same folder, including:

- `Nodule_Tool.py`
- `nodule_show/` folder
- Any other supporting files

## 🚀 How to Run
Once everything is in the same directory and dependencies are installed, use the following command to launch the app:
```bash
python.exe -m streamlit run Nodule_Tool.py
```
⚠️ Make sure you are using the correct Python environment (conda or venv) that has the required packages installed.

❗ Notes
Large libraries such as PyTorch binaries are not included in this repository due to GitHub file size limits.
Please install them via pip using the requirements.txt.

## 📜 License
This project is for **non-commercial use only**, such as academic projects or competitions.
![下載 (2)](https://github.com/user-attachments/assets/5d50c434-58b0-4872-bfd1-aa5e32c67853)

