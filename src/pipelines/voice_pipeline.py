import io

import librosa
import numpy as np
import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav


@st.cache_resource
def load_voice_encoder():
    """
    Load and cache the Resemblyzer VoiceEncoder.
    """
    return VoiceEncoder()


def get_voice_embedding(audio_bytes):
    """
    Convert uploaded audio bytes into a voice embedding.
    """
    try:
        encoder = load_voice_encoder()

        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        wav = preprocess_wav(audio)
        embedding = encoder.embed_utterance(wav)

        return embedding.tolist()

    except Exception as error:
        st.error(f"Voice recognition error: {error}")
        return None


def identify_speaker(
    new_embedding,
    candidates_dict,
    threshold=0.65
):
    """
    Identify the closest matching speaker.

    Returns:
        tuple: (speaker_id, similarity_score)
    """
    if new_embedding is None or not candidates_dict:
        return None, 0.0

    new_embedding = np.asarray(
        new_embedding,
        dtype=np.float32
    )

    best_sid = None
    best_score = -1.0

    for sid, stored_embedding in candidates_dict.items():

        if stored_embedding is None:
            continue

        try:
            stored_embedding = np.asarray(
                stored_embedding,
                dtype=np.float32
            )

            if new_embedding.shape != stored_embedding.shape:
                continue

            similarity = float(
                np.dot(
                    new_embedding,
                    stored_embedding
                )
            )

            if similarity > best_score:
                best_score = similarity
                best_sid = sid

        except (ValueError, TypeError):
            continue

    if best_score >= threshold:
        return best_sid, best_score

    return None, best_score


def process_bulk_audio(
    audio_bytes,
    candidates_dict,
    threshold=0.65
):
    """
    Process bulk audio and identify registered speakers.
    """
    try:
        encoder = load_voice_encoder()

        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            mono=True
        )

        segments = librosa.effects.split(
            audio,
            top_db=30
        )

        identified_results = {}

        for start, end in segments:

            if (end - start) < sample_rate * 0.5:
                continue

            segment_audio = audio[start:end]
            wav = preprocess_wav(segment_audio)

            embedding = encoder.embed_utterance(wav)

            sid, score = identify_speaker(
                embedding,
                candidates_dict,
                threshold
            )

            if sid is not None:
                if (
                    sid not in identified_results
                    or score > identified_results[sid]
                ):
                    identified_results[sid] = score

        return identified_results

    except Exception as error:
        st.error(f"Bulk audio processing error: {error}")
        return {}