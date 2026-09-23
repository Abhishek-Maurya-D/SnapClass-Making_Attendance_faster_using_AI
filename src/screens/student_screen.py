import streamlit as st
import time
import numpy as np

from PIL import Image

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout
)

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard

from src.pipelines.face_pipeline import (
    predict_attendance,
    get_face_embeddings,
    train_classifier
)

from src.pipelines.voice_pipeline import get_voice_embedding

from src.database.db import (
    get_all_students,
    create_student,
    get_student_subjects,
    get_student_attendance,
    unenroll_student_to_subject
)

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


# ============================================================
# STUDENT DASHBOARD
# ============================================================

def student_dashboard():

    student_data = st.session_state.student_data
    student_id = student_data["student_id"]

    # --------------------------------------------------------
    # Header + Logout
    # --------------------------------------------------------

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:
        header_dashboard()

    with c2:

        st.subheader(
            f"Welcome, {student_data['name']}"
        )

        if st.button(
            "Logout",
            type="secondary",
            key="student_logout_btn",
            shortcut="control+backspace"
        ):

            st.session_state["is_logged_in"] = False
            st.session_state["user_role"] = None

            if "student_data" in st.session_state:
                del st.session_state["student_data"]

            if "show_registration" in st.session_state:
                st.session_state["show_registration"] = False

            st.rerun()

    st.space()

    # --------------------------------------------------------
    # Enrolled Subjects
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        st.header("Your Enrolled Subjects")

    with c2:

        if st.button(
            "Enroll in Subject",
            type="primary",
            width="stretch"
        ):
            enroll_dialog()

    st.divider()

    # --------------------------------------------------------
    # Get Student Subjects & Attendance
    # --------------------------------------------------------

    with st.spinner(
        "Loading your enrolled subjects.."
    ):

        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    # --------------------------------------------------------
    # Attendance Statistics
    # --------------------------------------------------------

    stats_map = {}

    for log in logs:

        sid = log["subject_id"]

        if sid not in stats_map:

            stats_map[sid] = {
                "total": 0,
                "attended": 0
            }

        stats_map[sid]["total"] += 1

        if log.get("is_present"):

            stats_map[sid]["attended"] += 1

    # --------------------------------------------------------
    # Display Subjects
    # --------------------------------------------------------

    cols = st.columns(2)

    for i, sub_node in enumerate(subjects):

        sub = sub_node["subjects"]

        sid = sub["subject_id"]

        stats = stats_map.get(
            sid,
            {
                "total": 0,
                "attended": 0
            }
        )

        def unenroll_button():

            if st.button(
                "Unenroll from this course",
                type="tertiary",
                width="stretch",
                icon=":material/delete_forever:"
            ):

                unenroll_student_to_subject(
                    student_id,
                    sid
                )

                st.toast(
                    f"Unenrolled from {sub['name']} successfully!"
                )

                st.rerun()

        with cols[i % 2]:

            subject_card(
                name=sub["name"],
                code=sub["subject_code"],
                section=sub["section"],
                stats=[
                    (
                        "📅",
                        "Total",
                        stats["total"]
                    ),
                    (
                        "✅",
                        "Attended",
                        stats["attended"]
                    )
                ],
                footer_callback=unenroll_button
            )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    footer_dashboard()


# ============================================================
# STUDENT LOGIN / REGISTRATION SCREEN
# ============================================================

def student_screen():

    # --------------------------------------------------------
    # Page Styling
    # --------------------------------------------------------

    style_background_dashboard()
    style_base_layout()

    # --------------------------------------------------------
    # Initialize Registration State
    # --------------------------------------------------------

    if "show_registration" not in st.session_state:

        st.session_state["show_registration"] = False

    # --------------------------------------------------------
    # If Already Logged In
    # --------------------------------------------------------

    if "student_data" in st.session_state:

        student_dashboard()

        return

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    c1, c2 = st.columns(
        2,
        vertical_alignment="center",
        gap="xxlarge"
    )

    with c1:

        header_dashboard()

    with c2:

        if st.button(
            "Go back to Home",
            type="secondary",
            key="student_home_btn",
            shortcut="control+backspace"
        ):

            st.session_state["login_type"] = None
            st.session_state["show_registration"] = False

            st.rerun()

    # --------------------------------------------------------
    # Login Title
    # --------------------------------------------------------

    st.header(
        "Login using FaceID",
        text_alignment="center"
    )

    st.space()
    st.space()

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    photo_source = st.camera_input(
        "Position your face in the center"
    )

    # ========================================================
    # FACE LOGIN
    # ========================================================

    if photo_source:

        try:

            img = np.array(
                Image.open(photo_source)
            )

        except Exception as e:

            st.error(
                f"Unable to read captured image: {e}"
            )

            return

        # ----------------------------------------------------
        # Face Recognition
        # ----------------------------------------------------

        with st.spinner("AI is scanning.."):

            try:

                detected, all_ids, num_faces = predict_attendance(
                    img
                )

            except Exception as e:

                st.error(
                    f"Face recognition failed: {e}"
                )

                return

        # ====================================================
        # NO FACE
        # ====================================================

        if num_faces == 0:

            st.warning(
                "Face not found! Please position your face "
                "properly and take the photo again."
            )

            st.session_state["show_registration"] = False

        # ====================================================
        # MULTIPLE FACES
        # ====================================================

        elif num_faces > 1:

            st.warning(
                "Multiple faces found. Please make sure "
                "only one person is visible."
            )

            st.session_state["show_registration"] = False

        # ====================================================
        # SINGLE FACE
        # ====================================================

        else:

            # ------------------------------------------------
            # Face recognized by ML model
            # ------------------------------------------------

            if detected:

                try:

                    student_id = list(
                        detected.keys()
                    )[0]

                except (IndexError, AttributeError):

                    student_id = None

                # --------------------------------------------
                # Invalid Prediction
                # --------------------------------------------

                if not student_id:

                    st.warning(
                        "Unable to identify the student."
                    )

                    st.session_state[
                        "show_registration"
                    ] = True

                else:

                    # ----------------------------------------
                    # Search Database
                    # ----------------------------------------

                    all_students = get_all_students()

                    student = next(
                        (
                            s
                            for s in all_students
                            if s.get("student_id") == student_id
                        ),
                        None
                    )

                    # ----------------------------------------
                    # Student Exists
                    # ----------------------------------------

                    if student:

                        st.session_state[
                            "is_logged_in"
                        ] = True

                        st.session_state[
                            "user_role"
                        ] = "student"

                        st.session_state[
                            "student_data"
                        ] = student

                        st.session_state[
                            "show_registration"
                        ] = False

                        st.toast(
                            f"Welcome Back {student['name']}"
                        )

                        time.sleep(1)

                        st.rerun()

                    # ----------------------------------------
                    # Face Detected But Student Not Found
                    # ----------------------------------------

                    else:

                        st.warning(
                            "Face detected, but no student "
                            "profile was found."
                        )

                        st.info(
                            "You can register as a new student."
                        )

                        st.session_state[
                            "show_registration"
                        ] = True

            # ------------------------------------------------
            # Face NOT recognized
            # ------------------------------------------------

            else:

                st.info(
                    "Face not recognized! "
                    "You might be a new student."
                )

                st.session_state[
                    "show_registration"
                ] = True

    # ========================================================
    # NEW STUDENT REGISTRATION
    # ========================================================

    if st.session_state["show_registration"]:

        st.divider()

        with st.container(border=True):

            st.header(
                "Register New Profile"
            )

            # ------------------------------------------------
            # Name
            # ------------------------------------------------

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Hamza Rizvi",
                key="new_student_name"
            )

            # ------------------------------------------------
            # Voice Enrollment
            # ------------------------------------------------

            st.subheader(
                "Optional: Voice Enrollment"
            )

            st.info(
                "Enroll your voice for voice-only attendance."
            )

            audio_data = None

            try:

                audio_data = st.audio_input(
                    "Record a short phrase like "
                    "I am present, My name is Akash."
                )

            except Exception as e:

                st.warning(
                    f"Audio recording is unavailable: {e}"
                )

            # ------------------------------------------------
            # Create Account
            # ------------------------------------------------

            if st.button(
                "Create Account",
                type="primary",
                width="stretch",
                key="create_student_account"
            ):

                # --------------------------------------------
                # Validate Name
                # --------------------------------------------

                if not new_name.strip():

                    st.warning(
                        "Please enter your name!"
                    )

                # --------------------------------------------
                # Validate Photo
                # --------------------------------------------

                elif photo_source is None:

                    st.warning(
                        "Please capture your face first."
                    )

                else:

                    with st.spinner(
                        "Creating profile.."
                    ):

                        try:

                            # --------------------------------
                            # Convert Image
                            # --------------------------------

                            img = np.array(
                                Image.open(photo_source)
                            )

                            # --------------------------------
                            # Get Face Embedding
                            # --------------------------------

                            encodings = get_face_embeddings(
                                img
                            )

                            if not encodings:

                                st.error(
                                    "Couldn't capture your "
                                    "facial features for registration."
                                )

                            else:

                                face_emb = encodings[0].tolist()

                                # ----------------------------
                                # Voice Embedding
                                # ----------------------------

                                voice_emb = None

                                if audio_data:

                                    try:

                                        voice_emb = get_voice_embedding(
                                            audio_data.read()
                                        )

                                    except Exception as e:

                                        st.warning(
                                            "Voice enrollment failed. "
                                            "The account will be created "
                                            "without voice enrollment."
                                        )

                                        voice_emb = None

                                # ----------------------------
                                # Create Student
                                # ----------------------------

                                response_data = create_student(
                                    new_name.strip(),
                                    face_embedding=face_emb,
                                    voice_embedding=voice_emb
                                )

                                # ----------------------------
                                # Account Created
                                # ----------------------------

                                if response_data:

                                    # Train face classifier
                                    try:

                                        train_classifier()

                                    except Exception as e:

                                        st.warning(
                                            f"Profile created, but "
                                            f"classifier training failed: {e}"
                                        )

                                    # Get newly created student
                                    new_student = response_data[0]

                                    st.session_state[
                                        "is_logged_in"
                                    ] = True

                                    st.session_state[
                                        "user_role"
                                    ] = "student"

                                    st.session_state[
                                        "student_data"
                                    ] = new_student

                                    st.session_state[
                                        "show_registration"
                                    ] = False

                                    st.toast(
                                        f"Profile Created! "
                                        f"Hi {new_name.strip()}!"
                                    )

                                    time.sleep(1)

                                    st.rerun()

                                else:

                                    st.error(
                                        "Unable to create the "
                                        "student profile. Please try again."
                                    )

                        except Exception as e:

                            st.error(
                                f"Registration failed: {e}"
                            )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    footer_dashboard()