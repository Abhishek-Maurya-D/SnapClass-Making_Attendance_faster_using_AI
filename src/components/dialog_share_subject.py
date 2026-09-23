import io

import segno
import streamlit as st


@st.dialog("Share Class Link")
def share_subject_dialog(subject_name, subject_code):
    app_domain = "snapclass-attendance-mains.streamlit.app"

    join_url = f"{app_domain}/?join-code={subject_code}"

    st.header("Scan to Join")

    # Generate QR code
    qr = segno.make(join_url)

    output = io.BytesIO()

    qr.save(
        output,
        kind="png",
        scale=10,
        border=1,
    )

    col1, col2 = st.columns(2)

    # Copy link section
    with col1:
        st.markdown("### Copy Link")

        st.code(join_url, language="text")
        st.code(subject_code, language="text")

        st.info(
            "Copy this link to share on WhatsApp or Email."
        )

    # QR code section
    with col2:
        st.markdown("### Scan to Join")

        st.image(
            output.getvalue(),
            caption="QR Code for class joining",
        )