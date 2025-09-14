import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from PIL import Image, ImageTk
import sqlite3
import csv
import hashlib  # Still used for IDs
import dropbox
from flask import Flask, render_template, request, redirect, url_for
import time
import bcrypt  # For password hashing
import queue  # Added for notification queue

# Dropbox setup - Use OAuth 2.0 with refresh token
APP_KEY = "o53lt4n3vj8gmga"
APP_SECRET = "7pb540zfk67b45c"
REFRESH_TOKEN = "FlfSA2x9SI0AAAAAAAAAAYvn7ZmLr5hozPtIC-vOUQoSkUGlwFS0BtR8l45dNbnK"

dbx = dropbox.Dropbox(
    oauth2_refresh_token=REFRESH_TOKEN,
    app_key=APP_KEY,
    app_secret=APP_SECRET
)

try:
    account = dbx.users_get_current_account()
    print(f"Connected to Dropbox account: {account.email}")
    try:
        dbx.files_create_folder_v2("/Attendance")
        print("Created /Attendance folder")
    except dropbox.exceptions.ApiError as e:
        if e.error.is_path() and e.error.get_path().is_conflict():
            print("/Attendance folder already exists")
        else:
            raise
except dropbox.exceptions.AuthError as e:
    print(f"Authentication failed: {e}")
    exit(1)

ATTENDANCE_DB = "attendance.db"
FACE_TRAINED_MODEL = "face_trained.yml"

def init_db():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id TEXT PRIMARY KEY, name TEXT NOT NULL, roll_no TEXT NOT NULL, date TEXT NOT NULL, time TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS face_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, roll_no TEXT NOT NULL, face_encoding BLOB NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS name_to_id
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS allowed_students
                 (name TEXT NOT NULL, roll_no TEXT NOT NULL, UNIQUE(name, roll_no))''')
    conn.commit()
    conn.close()

def sync_allowed_students_from_cloud():
    dropbox_path = "/Attendance/allowed_students.csv"
    local_path = "allowed_students.csv"
    try:
        print(f"Downloading {dropbox_path} from Dropbox...")
        dbx.files_download_to_file(local_path, dropbox_path)
        print(f"Downloaded {local_path} successfully.")
        conn = sqlite3.connect(ATTENDANCE_DB)
        c = conn.cursor()
        c.execute('DELETE FROM allowed_students')
        with open(local_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            required_columns = {'sr_no', 'name', 'roll_no'}
            if not {'name', 'roll_no'}.issubset(csv_reader.fieldnames):
                missing = {'name', 'roll_no'} - set(csv_reader.fieldnames)
                raise ValueError(f"Missing required columns in CSV: {missing}")
            for row in csv_reader:
                name = row['name'].strip()
                roll_no = row['roll_no'].strip()
                sr_no = row.get('sr_no', 'N/A')
                print(f"Inserting: Sr No: {sr_no}, Name: {name}, Roll No: {roll_no}")
                c.execute('INSERT OR IGNORE INTO allowed_students (name, roll_no) VALUES (?, ?)', (name, roll_no))
        conn.commit()
        conn.close()
        print("Allowed students table updated from cloud data.")
        os.remove(local_path)
        print(f"Cleaned up local file: {local_path}")
    except dropbox.exceptions.ApiError as e:
        print(f"Dropbox ApiError: {e}. Using existing local data (if any).")
    except FileNotFoundError as e:
        print(f"Error: {e}. Could not find or process the local file.")
    except ValueError as e:
        print(f"CSV format error: {e}")
    except KeyError as e:
        print(f"KeyError: Missing expected column {e} in CSV file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def is_allowed_student(name, roll_no):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM allowed_students WHERE name = ? AND roll_no = ?', (name, roll_no))
    count = c.fetchone()[0]
    conn.close()
    return count > 0

def load_face_data():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT name, roll_no, face_encoding FROM face_data')
    face_db = {}
    for name, roll_no, face_encoding in c.fetchall():
        faces = np.frombuffer(face_encoding, dtype=np.float32).reshape(-1, 200, 200)
        face_db[name] = {"roll_no": roll_no, "faces": list(faces)}
    conn.close()
    return face_db

def save_face_data(name, roll_no, faces):
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    face_encoding = np.array(faces, dtype=np.float32).tobytes()
    c.execute('DELETE FROM face_data WHERE name = ?', (name,))
    c.execute('INSERT INTO face_data (name, roll_no, face_encoding) VALUES (?, ?, ?)', (name, roll_no, face_encoding))
    conn.commit()
    conn.close()

face_db = load_face_data()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=70)
last_attendance_time = {}

# New NotificationManager class with queue
notification_queue = queue.Queue()

class NotificationManager:
    def __init__(self, root):
        self.root = root
        self.current_window = None
        self.is_showing = False
        self.root.after(100, self.check_queue)  # Periodically check the queue

    def show_notification(self, title, message):
        notification_queue.put((title, message))

    def check_queue(self):
        try:
            while True:
                title, message = notification_queue.get_nowait()
                self.display_notification(title, message)
        except queue.Empty:
            pass
        self.root.after(100, self.check_queue)  # Schedule the next check

    def display_notification(self, title, message):
        if self.is_showing:
            return
        window = tk.Toplevel(self.root)
        window.title("")
        window.geometry("300x150+50+50")
        window.configure(bg='#2c3e50')
        window.overrideredirect(True)
        window.lift()
        window.attributes('-topmost', True)
        frame = ttk.Frame(window)
        frame.pack(padx=10, pady=10, fill='both', expand=True)
        title_label = ttk.Label(frame, text=title, font=('Helvetica', 12, 'bold'), wraplength=280)
        title_label.pack(pady=(0, 5))
        msg_label = ttk.Label(frame, text=message, wraplength=280)
        msg_label.pack(pady=5)
        def on_dismiss():
            self.is_showing = False
            window.destroy()
        close_btn = ttk.Button(frame, text="Dismiss", command=on_dismiss)
        close_btn.pack(pady=5)
        self.current_window = window
        self.is_showing = True

def show_notification(title, message):
    notification_manager.show_notification(title, message)

def mark_attendance(name, roll_no):
    current_time = datetime.now()
    if name in last_attendance_time:
        time_diff = current_time - last_attendance_time[name]
        if time_diff.total_seconds() < 30:
            print(f"Skipping attendance for {name}: too soon since last mark")
            return "Attendance skipped: Too soon since last mark."
    
    date_str = current_time.strftime("%Y-%m-%d")
    time_str = current_time.strftime("%H:%M:%S")
    unique_id = hashlib.md5(f"{roll_no}{date_str}{time_str}".encode()).hexdigest()[:10]
    
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('''SELECT time FROM attendance WHERE roll_no = ? AND date = ?''', (roll_no, date_str))
    existing_record = c.fetchone()
    
    if existing_record:
        print(f"Attendance already marked for {name} at {existing_record[0]}")
        conn.close()
        return f"Attendance already marked for {name} at {existing_record[0]}."
    
    try:
        c.execute('''INSERT INTO attendance (id, name, roll_no, date, time)
                     VALUES (?, ?, ?, ?, ?)''', (unique_id, name, roll_no, date_str, time_str))
        conn.commit()
        print(f"Attendance marked for {name} (Roll No: {roll_no}) at {time_str}")
        
        csv_file = f"attendance_{date_str}.csv"
        file_exists = os.path.isfile(csv_file)
        sr_no = 1
        if file_exists:
            with open(csv_file, 'r') as f:
                sr_no = sum(1 for line in f)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Sr No", "ID", "Name", "Roll No", "Date", "Time"])
            writer.writerow([sr_no, unique_id, name, roll_no, date_str, time_str])
        print(f"Saved to {csv_file} with Sr No: {sr_no}")
        
        dropbox_path = f"/Attendance/{csv_file}"
        link = None
        try:
            if not os.path.exists(csv_file):
                print(f"Error: {csv_file} does not exist locally before upload")
                return "Error: Attendance file not found locally."
            with open(csv_file, 'rb') as f:
                file_size = os.path.getsize(csv_file)
                print(f"Uploading {csv_file} with size {file_size} bytes")
                dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
            print(f"Uploaded {csv_file} to Dropbox at {dropbox_path}")
            link = dbx.sharing_create_shared_link(dropbox_path).url
            print(f"Shareable link: {link}")
        except dropbox.exceptions.ApiError as e:
            print(f"Dropbox ApiError during upload: {e}")
            return f"Error uploading to Dropbox: {str(e)}"
        except Exception as e:
            print(f"Unexpected error during upload: {e}")
            return f"Error during upload: {str(e)}"
        
        last_attendance_time[name] = current_time
        return f"Attendance marked for {name} (Roll No: {roll_no}) at {time_str}. Link: {link}"
    except sqlite3.IntegrityError as e:
        print(f"IntegrityError: {e}")
        return f"Error marking attendance for {name}."
    except Exception as e:
        print(f"Unexpected error in mark_attendance: {e}")
        raise
    finally:
        conn.close()

def train_model():
    labels = []
    face_samples = []
    if not face_db:
        return
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('DELETE FROM name_to_id')
    for i, name in enumerate(face_db.keys()):
        c.execute('INSERT INTO name_to_id (name) VALUES (?)', (name,))
        c.execute('SELECT id FROM name_to_id WHERE name = ?', (name,))
        person_id = c.fetchone()[0]
        for face in face_db[name]["faces"]:
            face_samples.append(face)
            labels.append(person_id)
    conn.commit()
    conn.close()
    if face_samples and labels:
        recognizer.train(face_samples, np.array(labels, dtype=np.int32))
        recognizer.write(FACE_TRAINED_MODEL)
    else:
        print("Training skipped: No faces found!")

def add_face(name, roll_no):
    if not is_allowed_student(name, roll_no):
        print(f"Unauthorized: {name} ({roll_no})")
        show_notification("Not Authorized", f"{name} (Roll No: {roll_no}) is not in the allowed students list.")
        return
    
    print(f"Opening camera for {name} ({roll_no})")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        show_notification("Camera Error", "Could not access the camera. Check permissions or connection.")
        return
    
    count = 0
    face_data = []
    start_time = time.time()
    timeout_seconds = 60  # 1 minute timeout
    
    while count < 20:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print("Face capture timed out after 1 minute.")
            cap.release()
            cv2.destroyAllWindows()
            show_notification("Timeout", "Face capture timed out after 1 minute.")
            return
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read camera frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            if os.path.exists(FACE_TRAINED_MODEL):
                is_similar, existing_name = check_face_similarity(face_resized)
                if is_similar:
                    existing_roll_no = face_db.get(existing_name, {}).get("roll_no", "Unknown")
                    if existing_name == name and existing_roll_no == roll_no:
                        cv2.putText(frame, "Face already registered", (50, frame.shape[0] - 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.imshow("Face Capture", frame)
                        cv2.waitKey(2000)
                        cap.release()
                        cv2.destroyAllWindows()
                        show_notification("Duplicate Detected", 
                                        f"Your face is already registered under this name: {name} and roll number: {roll_no}")
                        return
                    else:
                        cv2.putText(frame, f"Similar to: {existing_name}", (50, frame.shape[0] - 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        cv2.imshow("Face Capture", frame)
                        cv2.waitKey(2000)
                        cap.release()
                        cv2.destroyAllWindows()
                        show_notification("Duplicate Detected", 
                                        f"Your face is similar to an existing registration under {existing_name} (Roll No: {existing_roll_no})")
                        return
            face_data.append(face_resized)
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing {count}/20 - Time left: {int(timeout_seconds - elapsed_time)}s", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if count == 20:
        face_db[name] = {"roll_no": roll_no, "faces": face_data}
        save_face_data(name, roll_no, face_data)
        train_model()
        show_notification("Face Added", f"Successfully added {name} (Roll No: {roll_no})")
    else:
        show_notification("Error", "Not enough face data captured within 1 minute. Try again.")

def check_face_similarity(new_face, threshold=70):
    if not face_db or not os.path.exists(FACE_TRAINED_MODEL):
        return False, None
    recognizer.read(FACE_TRAINED_MODEL)
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT id, name FROM name_to_id')
    id_to_name = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    label, confidence = recognizer.predict(new_face)
    if confidence < threshold:
        return True, id_to_name.get(label, "Unknown")
    return False, None

def recognize_faces():
    if not os.path.exists(FACE_TRAINED_MODEL):
        print("Model not found, retraining...")
        train_model()
    recognizer.read(FACE_TRAINED_MODEL)
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT id, name FROM name_to_id')
    id_to_name = {row[0]: row[1] for row in c.fetchall()}
    conn.close()
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return "Error: Could not open webcam."
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", 640, 480)
    
    consecutive_matches = {}
    required_matches = 5  # Correct variable name
    confidence_threshold = 60
    
    start_time = time.time()
    timeout_seconds = 60
    no_face_counter = 0
    no_face_threshold = 50
    
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print("Recognition timed out after 1 minute.")
            cap.release()
            cv2.destroyAllWindows()
            return "Recognition timed out after 1 minute."
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera frame could not be captured")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        if len(faces) == 0:
            no_face_counter += 1
            if no_face_counter >= no_face_threshold:
                cv2.putText(frame, "No face detected - Check lighting", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            no_face_counter = 0
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (200, 200))
            label, confidence = recognizer.predict(face_resized)
            if confidence < confidence_threshold:
                name = id_to_name.get(label, "Unknown")
                if name != "Unknown":
                    roll_no = face_db.get(name, {}).get("roll_no", "Unknown")
                    if is_allowed_student(name, roll_no):
                        consecutive_matches[name] = consecutive_matches.get(name, 0) + 1
                        text = f"{name} ({confidence:.2f}) - Verifying ({consecutive_matches[name]}/{required_matches})"
                        color = (0, 255, 0)
                        if consecutive_matches[name] >= required_matches:  # Fixed typo here
                            result = mark_attendance(name, roll_no)
                            print(f"Attendance marked for {name}")
                            cv2.waitKey(1000)
                            cap.release()
                            cv2.destroyAllWindows()
                            return result
                    else:
                        text = "Not Authorized"
                        color = (0, 0, 255)
                        consecutive_matches.clear()
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
                    consecutive_matches.clear()
            else:
                text = f"Low Confidence ({confidence:.2f})"
                color = (0, 0, 255)
                consecutive_matches.clear()
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        remaining_time = int(timeout_seconds - elapsed_time)
        cv2.putText(frame, f"Time left: {remaining_time}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return "Recognition stopped manually."

app = Flask("Attendance Marking System")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_face', methods=['GET', 'POST'])
def add_face_web():
    if request.method == 'POST':
        print("POST request received for /add_face")
        name = request.form['name']
        roll_no = request.form['roll_no']
        print(f"Received: Name={name}, Roll No={roll_no}")
        if name and roll_no:
            if not is_allowed_student(name, roll_no):
                print(f"Student {name} ({roll_no}) not in allowed_students")
                return render_template('add_face.html', error="You are not in the allowed students list.")
            print(f"Starting face capture for {name} ({roll_no})")
            threading.Thread(target=add_face, args=(name, roll_no)).start()
            return render_template('add_face.html', message="Face capture started. Please check the camera window.")
        else:
            print("Missing name or roll_no")
            return render_template('add_face.html', error="Name and Roll No are required.")
    print("GET request for /add_face")
    return render_template('add_face.html')

@app.route('/start_recognition')
def start_recognition_web():
    result = recognize_faces()
    return render_template('index.html', message=result)

@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance_web():
    date = request.form.get('date', datetime.now().strftime("%Y-%m-%d"))
    print(f"Viewing attendance for date: {date}")
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('''SELECT name, roll_no, time FROM attendance WHERE date = ? ORDER BY time''', (date,))
    records = c.fetchall()
    print(f"Found {len(records)} records: {records}")
    conn.close()
    
    csv_file = f"attendance_{date}.csv"
    dropbox_path = f"/Attendance/{csv_file}"
    drive_link = None
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sr No", "ID", "Name", "Roll No", "Date", "Time"])
        for i, record in enumerate(records, 1):
            unique_id = hashlib.md5(f"{record[1]}{date}{record[2]}".encode()).hexdigest()[:10]
            writer.writerow([i, unique_id, record[0], record[1], date, record[2]])
    print(f"Regenerated {csv_file} with {len(records)} records")
    
    try:
        with open(csv_file, 'rb') as f:
            print(f"Uploading {csv_file} to Dropbox")
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
        drive_link = dbx.sharing_create_shared_link(dropbox_path).url
        print(f"Generated Dropbox link: {drive_link}")
    except dropbox.exceptions.ApiError as e:
        print(f"Dropbox ApiError in view_attendance: {e}")
        drive_link = "Unable to generate link due to Dropbox error"
    
    return render_template('view_attendance.html', records=records, date=date, drive_link=drive_link)

def cleanup_face_data():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT name FROM allowed_students')
    allowed_names = set(row[0] for row in c.fetchall())
    c.execute('SELECT name FROM face_data')
    face_data_names = set(row[0] for row in c.fetchall())
    names_to_remove = face_data_names - allowed_names
    for name in names_to_remove:
        c.execute('DELETE FROM face_data WHERE name = ?', (name,))
        print(f"Removed face data for {name} as they are no longer in allowed_students.")
    conn.commit()
    conn.close()
    if names_to_remove:
        global face_db
        face_db = load_face_data()
        train_model()

@app.route('/sync_students', methods=['GET', 'POST'])
def sync_students_web():
    print("Sync route accessed!")
    if request.method == 'POST':
        try:
            sync_allowed_students_from_cloud()
            cleanup_face_data()
            return render_template('sync_students.html', message="Successfully synced allowed students from Dropbox.")
        except Exception as e:
            return render_template('sync_students.html', message=f"Error syncing: {str(e)}")
    return render_template('sync_students.html', message=None)

# Precomputed bcrypt hash of "admin123" (replace with your generated hash)
ADMIN_PASSWORD_HASH = b'$2b$12$h69.OU4ClQVc5cu347lYi.qmwOgQ8zNa02e2cnse.5rLBgZ7ElDjW'  # Replace with hash from bcrypt generation

@app.route('/add_student', methods=['GET', 'POST'])
def add_student_web():
    if request.method == 'POST':
        password = request.form.get('password', '').strip()
        name = request.form.get('name')
        roll_no = request.form.get('roll_no')
        
        if not (name and roll_no and password):
            return render_template('add_student.html', error="All fields are required.", 
                                   name=name, roll_no=roll_no)
        
        # Verify password with bcrypt
        try:
            if not bcrypt.checkpw(password.encode(), ADMIN_PASSWORD_HASH):
                print(f"Password verification failed for: {repr(password)}")
                return render_template('add_student.html', error="Incorrect admin password.", 
                                       name=name, roll_no=roll_no)
        except Exception as e:
            print(f"Error verifying password: {e}")
            return render_template('add_student.html', error="Password verification failed.", 
                                   name=name, roll_no=roll_no)
        
        print("Password accepted!")
        
        try:
            conn = sqlite3.connect(ATTENDANCE_DB)
            c = conn.cursor()
            c.execute('INSERT OR IGNORE INTO allowed_students (name, roll_no) VALUES (?, ?)', (name, roll_no))
            conn.commit()
            conn.close()
            print(f"Inserted student: Name: {name}, Roll No: {roll_no}")
            
            update_allowed_students_csv()
            return render_template('add_student.html', message=f"Successfully added {name} (Roll No: {roll_no})", 
                                   name='', roll_no='')
        except sqlite3.IntegrityError:
            return render_template('add_student.html', error="This Name and Roll No combination already exists.", 
                                   name=name, roll_no=roll_no)
        except Exception as e:
            print(f"Error adding student: {e}")
            return render_template('add_student.html', error=f"Error: {str(e)}", 
                                   name=name, roll_no=roll_no)
    
    return render_template('add_student.html', message=None, error=None, name='', roll_no='')

def update_allowed_students_csv():
    conn = sqlite3.connect(ATTENDANCE_DB)
    c = conn.cursor()
    c.execute('SELECT name, roll_no FROM allowed_students')
    students = c.fetchall()
    conn.close()
    
    csv_file = "allowed_students.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sr_no", "name", "roll_no"])
        for i, (name, roll_no) in enumerate(students, 1):  # Auto-generate sr_no starting from 1
            writer.writerow([i, name, roll_no])
    print(f"Regenerated {csv_file} with {len(students)} students")
    
    dropbox_path = "/Attendance/allowed_students.csv"
    try:
        with open(csv_file, 'rb') as f:
            print(f"Uploading {csv_file} to Dropbox")
            dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode('overwrite'))
        print(f"Uploaded {csv_file} to Dropbox at {dropbox_path}")
    except dropbox.exceptions.ApiError as e:
        print(f"Dropbox ApiError during upload: {e}")
    finally:
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"Cleaned up local file: {csv_file}")

def main():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO allowed_students (name, roll_no) VALUES (?, ?)", ("Test Student", "TS001"))
    conn.commit()
    conn.close()
    print("Added Test Student (TS001) to allowed_students")
    init_db()
    
    # Create Tkinter root and notification manager
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    global notification_manager
    notification_manager = NotificationManager(root)
    
    # Run Flask in a separate thread with debug=False
    threading.Thread(target=lambda: app.run(debug=False, host='0.0.0.0', port=5000)).start()
    
    # Run Tkinter event loop in the main thread
    root.mainloop()

if __name__ == "__main__":
    main()