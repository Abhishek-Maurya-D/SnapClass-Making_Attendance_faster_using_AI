import dlib
import numpy as np
import face_recognition_models

from sklearn.svm import SVC

import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    shape_predictor = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    face_recognition_model = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, shape_predictor, face_recognition_model


def get_face_embeddings(image_np):
    detector, shape_predictor, face_recognition_model = (
        load_dlib_models()
    )

    faces = detector(image_np, 1)

    encodings = []

    for face in faces:
        shape = shape_predictor(image_np, face)

        face_descriptor = (
            face_recognition_model.compute_face_descriptor(
                image_np,
                shape,
                1
            )
        )

        encodings.append(
            np.array(face_descriptor)
        )

    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get("face_embedding")
        student_id = student.get("student_id")

        if embedding is None or student_id is None:
            continue

        try:
            embedding_array = np.asarray(
                embedding,
                dtype=np.float64
            )

            if embedding_array.shape != (128,):
                continue

            X.append(embedding_array)
            y.append(student_id)

        except (ValueError, TypeError):
            continue

    if not X:
        return None

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    unique_students = np.unique(y)

    if len(unique_students) == 1:
        return {
            "clf": None,
            "X": X,
            "y": y
        }

    clf = SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced"
    )

    try:
        clf.fit(X, y)

    except ValueError as error:
        st.error(
            f"Failed to train face classifier: {error}"
        )
        return None

    return {
        "clf": clf,
        "X": X,
        "y": y
    }


def train_classifier():
    st.cache_resource.clear()

    model_data = get_trained_model()

    return model_data is not None


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    model_data = get_trained_model()

    if model_data is None:
        return detected_student, [], len(encodings)

    clf = model_data["clf"]
    X_train = model_data["X"]
    y_train = model_data["y"]

    all_students = sorted(
        list(set(y_train.tolist()))
    )

    if not all_students:
        return detected_student, [], len(encodings)

    for encoding in encodings:
        if clf is None:
            predicted_id = all_students[0]

        else:
            try:
                predicted_id = clf.predict(
                    [encoding]
                )[0]

            except Exception as error:
                st.warning(
                    f"Could not classify a detected face: {error}"
                )
                continue

        matching_indices = np.where(
            y_train == predicted_id
        )[0]

        if len(matching_indices) == 0:
            continue

        best_match_score = float("inf")

        for index in matching_indices:
            student_embedding = X_train[index]

            distance = np.linalg.norm(
                student_embedding - encoding
            )

            best_match_score = min(
                best_match_score,
                distance
            )

        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:
            detected_student[int(predicted_id)] = True

    return (
        detected_student,
        all_students,
        len(encodings)
    )