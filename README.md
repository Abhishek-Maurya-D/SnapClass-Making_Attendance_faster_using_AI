# 📸 SnapClass — Making Attendance Faster Using AI

SnapClass is an AI-powered attendance system built with **Streamlit**. Teachers take attendance for an entire classroom in seconds by snapping a few photos or recording a short clip of students speaking — no roll call, no manual marking. Students get a passwordless experience too: they log in with their **face**.

## 🔗 Quick Links

| Link | Purpose |
|---|---|
| 🚀 **[Launch the App](https://snapclass-attendance-mains.streamlit.app/)** | Go straight to the deployed app and start using it |
| 📖 **[Landing Page / How to Use](https://snap-class-making-attendance-landin.vercel.app/)** | Learn about SnapClass and how to use it before you dive in |

---

## ✨ Features

### 👨‍🏫 For Teachers
- **Register / Login** with a username and password
- **Create subjects** (with a name, code, and section) and manage them from a dashboard
- **Share a subject** via a generated join link and **QR code** so students can enroll instantly by scanning it
- **Take attendance with AI** in two ways:
  - 📷 **Photo Attendance** — upload/capture multiple classroom photos; the AI detects every face and cross-checks it against enrolled students
  - 🎤 **Voice Attendance** — record students saying a phrase like *"I am present"*; the AI splits the recording into segments and matches each voice against enrolled students
- **Review results** before they're saved, then log attendance to the database
- **View attendance records**, grouped by session with present/total counts per class

### 👨‍🎓 For Students
- **Face login** — no passwords, just a webcam snapshot
- **Self-registration** on first login: capture a face photo and optionally record a voice sample for voice-based attendance
- **Enroll in subjects** using a subject code or by scanning a teacher's QR code / join link
- **Track attendance** for every enrolled subject (classes attended vs. total classes)
- **Unenroll** from a subject at any time

---

## 🧠 How the AI Works

**Face Recognition**
- Faces are detected and encoded into 128-dimensional vectors using **dlib**'s HOG face detector, shape predictor, and ResNet face-recognition model.
- An **SVM classifier** (scikit-learn) is trained on the fly from every enrolled student's stored face embedding.
- For a new classroom photo, each detected face is classified, then verified against the closest stored embedding using a Euclidean-distance threshold, so a student is only marked present when the match is confident.

**Voice Recognition**
- Voice samples are encoded into embeddings using **Resemblyzer**.
- A classroom recording is split into voiced segments with **Librosa**, and each segment's embedding is compared (cosine similarity) against enrolled students' stored voice embeddings.
- A student is marked present if their best match crosses a similarity threshold.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| App Framework | [Streamlit](https://streamlit.io/) |
| Language | Python |
| Database & Auth | [Supabase](https://supabase.com/) (PostgreSQL) |
| Face Recognition | dlib, `face_recognition_models`, scikit-learn (SVM) |
| Voice Recognition | Resemblyzer, Librosa |
| QR Codes | Segno |
| Password Security | bcrypt |
| Data Handling | NumPy, Pandas, Pillow |

---

## 📂 Project Structure

```
SnapClass-Making_Attendance_faster_using_AI/
├── app.py                     # App entry point & routing (home / student / teacher)
├── requirements.txt           # Python dependencies
└── src/
    ├── screens/
    │   ├── home_screen.py     # Landing page — choose Student or Teacher portal
    │   ├── student_screen.py  # Face login, self-registration, student dashboard
    │   └── teacher_screen.py  # Teacher login/register, attendance, subjects, records
    ├── pipelines/
    │   ├── face_pipeline.py   # Face detection, embeddings, SVM training & prediction
    │   └── voice_pipeline.py  # Voice embeddings, speaker identification, bulk audio
    ├── components/
    │   ├── dialog_create_subject.py     # Create a new subject
    │   ├── dialog_share_subject.py      # Share subject via link/QR code
    │   ├── dialog_enroll.py             # Student enrolls via subject code
    │   ├── dialog_auto_enroll.py        # Auto-enroll via join-code URL param
    │   ├── dialog_add_photo.py          # Add classroom photos for attendance
    │   ├── dialog_voice_attendance.py   # Record & analyze classroom audio
    │   ├── dialog_attendance_results.py # Preview & confirm attendance results
    │   ├── subject_card.py, header.py, footer.py
    ├── database/
    │   ├── config.py           # Supabase client setup
    │   └── db.py               # All database queries (students, teachers, subjects, attendance)
    └── ui/
        └── base_layout.py      # Shared page styling
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A [Supabase](https://supabase.com/) project (URL + API key) with tables for `teachers`, `students`, `subjects`, `subject_students`, and `attendance`

### 1. Clone the repository
```bash
git clone https://github.com/Abhishek-Maurya-D/SnapClass-Making_Attendance_faster_using_AI.git
cd SnapClass-Making_Attendance_faster_using_AI
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ `dlib-bin` and `face_recognition_models` may take a few minutes to install as they pull in compiled binaries.

### 4. Configure Supabase secrets
Create a `.streamlit/secrets.toml` file in the project root:
```toml
SUPABASE_URL = "your-supabase-project-url"
SUPABASE_KEY = "your-supabase-api-key"
```

### 5. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## 🖥️ Live Demo

- Try the deployed app here 👉 **[snapclass-attendance-mains.streamlit.app](https://snapclass-attendance-mains.streamlit.app/)**
- New to SnapClass? Check the usage guide first 👉 **[Landing Page](https://snap-class-making-attendance-landin.vercel.app/)**

---

## 🔮 Possible Future Enhancements
- Liveness detection to prevent photo/video spoofing during face login
- Downloadable attendance reports (Excel/CSV export)
- Push/email notifications for low attendance
- Mobile-optimized camera capture flow

---

## 👤 Author

**Abhishek Maurya**
[GitHub](https://github.com/Abhishek-Maurya-D) · [Repository](https://github.com/Abhishek-Maurya-D/SnapClass-Making_Attendance_faster_using_AI)

---

## 📄 License

No license file is currently included in this repository. Add one (e.g., MIT) if you intend for others to reuse this code.
