# attendance_logger.py
# This file handles saving attendance to a CSV file
# When a face is recognized, we call mark_attendance(name)
# It saves name, date, time and prevents duplicate entries

import pandas as pd
import os
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────
ATTENDANCE_FOLDER = "attendance"
ATTENDANCE_FILE = os.path.join(ATTENDANCE_FOLDER, "attendance.csv")

def mark_attendance(name):
    """
    This function is called when a face is recognized.
    It saves the name, date and time to the CSV file.
    It also prevents marking the same person twice on same day.
    """

    # get current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")   # example: 2026-05-21
    time = now.strftime("%H:%M:%S")   # example: 10:30:45

    # ── Check if CSV file already exists ─────────────────────
    if os.path.exists(ATTENDANCE_FILE):
        # load existing attendance
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        # create empty dataframe with columns
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # ── Check for duplicate ───────────────────────────────────
    # don't mark same person twice on same day
    already_marked = (
        (df["Name"] == name) &
        (df["Date"] == date)
    ).any()

    if already_marked:
        print(f"  {name} already marked present today!")
        return False

    # ── Add new attendance record ─────────────────────────────
    new_record = pd.DataFrame([{
        "Name": name,
        "Date": date,
        "Time": time
    }])

    df = pd.concat([df, new_record], ignore_index=True)

    # ── Save back to CSV ──────────────────────────────────────
    df.to_csv(ATTENDANCE_FILE, index=False)

    print(f"  Attendance marked for {name} at {time} on {date}")
    return True


def get_attendance():
    """
    Returns all attendance records as a dataframe
    Used later by the dashboard
    """
    if os.path.exists(ATTENDANCE_FILE):
        return pd.read_csv(ATTENDANCE_FILE)
    else:
        return pd.DataFrame(columns=["Name", "Date", "Time"])


# ── Test the logger ───────────────────────────────────────────
if __name__ == "__main__":
    print("Testing attendance logger...\n")

    # test marking attendance
    mark_attendance("divya")
    mark_attendance("divya")  # should say already marked!
    mark_attendance("divya")  # should say already marked!

    print("\nCurrent attendance records:")
    print(get_attendance())