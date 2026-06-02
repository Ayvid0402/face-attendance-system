# dashboard.py
# Streamlit web dashboard for attendance system
# Shows data from both CSV and SQLite database
# Run with: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import plotly.express as px

sys.path.append("src")
from database import get_attendance_db, get_today_attendance, get_all_persons, create_database

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="👤",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("👤 Real-Time Face Attendance System")
st.markdown("Automatic attendance tracking using Face Recognition + Deep Learning")
st.divider()

# ── Load data ─────────────────────────────────────────────────
# create database if not exists
create_database()

df = get_attendance_db()
today_df = get_today_attendance()
persons_df = get_all_persons()

today = datetime.now().strftime("%Y-%m-%d")

# ── Top metrics ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📋 Total Records", len(df))

with col2:
    st.metric("✅ Present Today", len(today_df))

with col3:
    st.metric("👥 Total People", len(persons_df))

with col4:
    avg_conf = round(df["confidence"].mean(), 1) if len(df) > 0 else 0
    st.metric("🎯 Avg Confidence", f"{avg_conf}%")

st.divider()

# ── Today's attendance ────────────────────────────────────────
st.subheader(f"✅ Today's Attendance — {today}")

if len(today_df) > 0:
    cols = st.columns(len(today_df))
    for i, (_, row) in enumerate(today_df.iterrows()):
        with cols[i]:
            st.success(f"✅ **{row['name'].capitalize()}**")
            st.write(f"🕐 {row['time']}")
            st.write(f"🎯 {row['confidence']}%")
else:
    st.warning("No attendance marked today yet!")
    st.info("Run webcam_attendance.py to start marking attendance!")

st.divider()

# ── Two columns layout ────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("📋 Attendance Records")

    if len(df) > 0:
        filter_name = st.selectbox(
            "Filter by Name",
            ["All"] + sorted(df["name"].unique().tolist())
        )

        if filter_name != "All":
            filtered_df = df[df["name"] == filter_name]
        else:
            filtered_df = df

        st.dataframe(
            filtered_df.sort_values("date", ascending=False),
            use_container_width=True
        )

        # download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="attendance.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records yet!")

with right:
    st.subheader("📊 Attendance Charts")

    if len(df) > 0:
        # daily attendance chart
        daily_count = df.groupby("date").size().reset_index(name="Count")
        fig = px.bar(
            daily_count,
            x="date",
            y="Count",
            title="Daily Attendance",
            color="Count",
            color_continuous_scale="greens"
        )
        st.plotly_chart(fig, use_container_width=True)

        # attendance by person
        person_count = df.groupby("name").size().reset_index(name="Days Present")
        fig2 = px.pie(
            person_count,
            values="Days Present",
            names="name",
            title="Attendance by Person"
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No data to show yet!")

st.divider()

# ── Registered persons ────────────────────────────────────────
st.subheader("👥 Registered Persons")

if len(persons_df) > 0:
    cols = st.columns(4)
    for i, (_, row) in enumerate(persons_df.iterrows()):
        with cols[i % 4]:
            st.info(f"👤 **{row['name'].capitalize()}**")
            st.caption(f"Registered: {row['registered'][:10]}")
else:
    st.info("No persons registered yet!")

st.divider()

# ── Unknown faces ─────────────────────────────────────────────
st.subheader("❓ Unknown Faces Detected")

unknown_folder = "unknown_faces"
unknown_files = []

if os.path.exists(unknown_folder):
    unknown_files = [f for f in os.listdir(unknown_folder)
                    if f.endswith('.jpg') or f.endswith('.png')]

if len(unknown_files) > 0:
    st.warning(f"{len(unknown_files)} unknown face(s) detected!")
    cols = st.columns(4)
    for i, filename in enumerate(unknown_files[:8]):
        with cols[i % 4]:
            img_path = os.path.join(unknown_folder, filename)
            st.image(img_path, caption=filename, width=150)
else:
    st.info("No unknown faces detected!")

st.divider()

# ── Model comparison section ──────────────────────────────────
st.subheader("🔬 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Haar Cascade")
    st.markdown("""
    - ⚡ **Speed:** 6.6 FPS
    - 🎯 **Accuracy:** Lower
    - ✅ **Pros:** Very fast
    - ❌ **Cons:** False positives
    """)

with col2:
    st.markdown("### HOG + dlib (Our Method)")
    st.markdown("""
    - ⚡ **Speed:** 3.0 FPS
    - 🎯 **Accuracy:** Higher
    - ✅ **Pros:** More accurate
    - ✅ **Our Choice:** Better for attendance!
    """)

st.divider()

# ── Refresh button ────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Refresh Data"):
        st.rerun()

with col2:
    st.caption("Face Attendance System — OpenCV + face_recognition + SQLite + Streamlit")