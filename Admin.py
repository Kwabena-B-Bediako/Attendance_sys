import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import cv2
import numpy as np
import os
import csv
import time
import pickle
from datetime import datetime
import face_recognition

# Ensure the 'data' directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

details_file = os.path.join(data_dir, 'user_details.pkl')
faces_file = os.path.join(data_dir, 'face_data.pkl')
attendance_dir = 'Attendance'


if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Function to register a user
def register_user(first_name, last_name, index_number):
    # Check if user has already been registered
    if os.path.isfile(details_file):
        with open(details_file, 'rb') as f:
            details = pickle.load(f)
        if any(user['index'] == index_number for user in details):
            messagebox.showerror("Error", "This index number is already registered.")
            return
    def capture_face():
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        captured_image = None

        while True:
            ret, frame = video.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image. Please check your webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (150, 150))
                captured_image = resized_img
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 3)
                break

            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if captured_image is not None:
                break

        video.release()
        cv2.destroyAllWindows()

        if captured_image is not None:
            try:
                face_encoding = face_recognition.face_encodings(captured_image)[0]
            except IndexError:
                messagebox.showerror("Error", "No face detected. Please try again.")
                return
            
            # Load existing face encodings
            if os.path.isfile(faces_file):
                with open(faces_file, 'rb') as f:
                    face_data = pickle.load(f)
                if any(face_recognition.compare_faces(face_data, face_encoding, tolerance=0.6)):
                    messagebox.showerror("Error", "This face is already registered.")
                    return

            user_details = {'first_name': first_name, 'last_name': last_name, 'index': index_number}
            if not os.path.isfile(details_file):
                details = [user_details]
            else:
                with open(details_file, 'rb') as f:
                    details = pickle.load(f)
                details.append(user_details)
            with open(details_file, 'wb') as f:
                pickle.dump(details, f)

            if not os.path.isfile(faces_file):
                face_data = [face_encoding]
            else:
                with open(faces_file, 'rb') as f:
                    face_data = pickle.load(f)
                face_data.append(face_encoding)
            with open(faces_file, 'wb') as f:
                pickle.dump(face_data, f)

            messagebox.showinfo("Success", "Face data and user details saved successfully.")
        else:
            messagebox.showerror("Error", "No face captured. Please try again.")

    capture_face()

# Function to mark attendance
def mark_attendance():
    def capture_and_mark(course_code):
        course_codes = []
        with open('course_codes.txt', 'r') as f:
            course_codes = [line.strip() for line in f.readlines() if line.strip()]

        if course_code not in course_codes:
            messagebox.showerror("Error", "Invalid course code selected.")
            return
        
        if not os.path.isfile(details_file) or not os.path.isfile(faces_file):
            messagebox.showerror("Error", "User details or face data files not found. Please register first.")
            return
        
        with open(details_file, 'rb') as f:
            user_details = pickle.load(f)

        with open(faces_file, 'rb') as f:
            face_data = pickle.load(f)
        face_data = np.array(face_data)

        if len(user_details) != face_data.shape[0]:
            messagebox.showerror("Error", "Mismatch between the number of user details and face data samples.")
            return

        known_face_encodings = []
        known_face_names = []

        for user, face_encoding in zip(user_details, face_data):
            try:
                known_face_encodings.append(face_encoding)
                known_face_names.append(user['index'])
            except Exception as e:
                print(f"Could not encode face for user {user['index']}: {e}")
                continue

        # Display a prompt message
        prompt_window = tk.Toplevel()
        prompt_window.title("Attention")
        prompt_window.geometry("350x150")
        prompt_window.resizable(False, False)
        tk.Label(prompt_window, text="Please look directly into the camera and wait...", font=("Helvetica", 12)).pack(pady=20)
        tk.Button(prompt_window, text="Proceed", command=prompt_window.destroy, font=("Helvetica", 12)).pack(pady=20)
        prompt_window.grab_set()
        prompt_window.wait_window()
        

        video = cv2.VideoCapture(0)

        tolerance = 0.5  # Tolerance variable
        COL_NAMES = ['FIRST_NAME', 'LAST_NAME', 'INDEX', 'TIME']

        while True:
            ret, frame = video.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image. Please check your webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_recognized = False

            for face_encoding in face_encodings:
                try:
                    match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                    if True in match:
                        best_match_index = match.index(True)
                        index_number = known_face_names[best_match_index]
                        ts = time.time()
                        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        file_path = os.path.join(attendance_dir, f"Attendance_{course_code}_{date}.csv")
                        exist = os.path.isfile(file_path)

                        # Check if the user is already marked present in the CSV file
                        if exist:
                            with open(file_path, 'r', newline='') as csvfile:
                                reader = csv.DictReader(csvfile)
                                if any(row['INDEX'] == index_number for row in reader):
                                    messagebox.showinfo("Info", f"User with index {index_number} has already been marked present.")
                                    video.release()
                                    cv2.destroyAllWindows()
                                    return

                        # Find the user detail by index
                        user_detail = next((user for user in user_details if user['index'] == index_number), None)

                        if user_detail:
                            attendance = [user_detail['first_name'], user_detail['last_name'], index_number, str(timestamp)]
                            

                            with open(file_path, "a", newline='') as csvfile:
                                writer = csv.writer(csvfile)
                                if not exist:
                                    writer.writerow(COL_NAMES)
                                writer.writerow(attendance)
                            messagebox.showinfo("Success", f"Attendance marked for {user_detail['first_name']} {user_detail['last_name']}")
                            video.release()
                            cv2.destroyAllWindows()
                            return
                except IndexError:
                    messagebox.showerror("Error", "No face detected. Please make sure you are registered in the system.")
                    video.release()
                    cv2.destroyAllWindows()
                    return
                
            if not face_recognized:
                messagebox.showerror("Error", "Face not recognized. Please make sure you are registered in the system.")
                video.release()
                cv2.destroyAllWindows()
                return

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    def select_course_code():
        course_window = tk.Toplevel()
        course_window.title("Select Course Code")
        course_window.geometry("400x200")
        course_window.resizable(False, False)

        tk.Label(course_window, text="Select Course Code:", font=("Helvetica", 12)).pack(pady=5)
        course_code_var = tk.StringVar(course_window)

        # Read course codes from the file
        course_codes = []
        with open('course_codes.txt', 'r') as f:
            course_codes = [line.strip() for line in f.readlines() if line.strip()]

        if not course_codes:
            messagebox.showerror("Error", "No course codes found in the file.")
            course_window.destroy()
            return

        course_code_var.set(course_codes[0])  # default value

        course_code_menu = ttk.Combobox(course_window, textvariable=course_code_var, values=course_codes, font=("Helvetica", 12))
        course_code_menu.pack(pady=5)

        def submit_course_code():
            course_code = course_code_var.get()
            course_window.destroy()
            capture_and_mark(course_code)

        tk.Button(course_window, text="Submit", font=("Helvetica", 12, "bold"), command=submit_course_code).pack(pady=20)

    select_course_code()


# Function to view all users
def view_all_users():
    if not os.path.isfile(details_file):
        messagebox.showerror("Error", "User details file not found.")
        return

    with open(details_file, 'rb') as f:
        details = pickle.load(f)

    view_window = tk.Toplevel()
    view_window.title("View All Students")
    view_window.geometry("1000x550")
    view_window.resizable(False, False)

    frame = ttk.Frame(view_window)
    frame.pack(fill='both', expand=True)

    tree = ttk.Treeview(frame)
    tree['columns'] = ('No', 'index_number', 'last_name', 'first_name')
    tree.heading('No', text='No')
    tree.heading('index_number', text='Index Number')
    tree.heading('last_name', text='Last Name')
    tree.heading('first_name', text='First Name')

    for i, user in enumerate(details, start=1):
        tree.insert('', 'end', values=(i, user['index'], user['last_name'], user['first_name']))

    

    # Vertical Scrollbar
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

    tree.pack(fill='both', expand=True)

    # Horizontal Scrollbar
    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    hsb.pack(side='bottom', fill='x')
    tree.configure(xscrollcommand=hsb.set)

    def delete_selected_user():
        selected_item = tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a user to delete.")
            return

        selected_user = tree.item(selected_item, 'values')
        index_number = selected_user[1]

        if not os.path.isfile(details_file) or not os.path.isfile(faces_file):
            messagebox.showerror("Error", "User details or face data files not found.")
            return

        with open(details_file, 'rb') as f:
            details = pickle.load(f)

        with open(faces_file, 'rb') as f:
            face_data = pickle.load(f)

        user_index = next((i for i, user in enumerate(details) if user['index'] == index_number), None)

        if user_index is not None:
            del details[user_index]
            del face_data[user_index]

            with open(details_file, 'wb') as f:
                pickle.dump(details, f)

            with open(faces_file, 'wb') as f:
                pickle.dump(face_data, f)

            messagebox.showinfo("Success", "User deleted successfully.")
            tree.delete(selected_item)
        else:
            messagebox.showerror("Error", "Index number not found.")

    delete_button = ttk.Button(view_window, text="Delete Selected User", command=delete_selected_user)
    delete_button.pack(pady=10)

def open_view():
        view_all_users()

#Function to add a course
def add_course_code(course_code):

    # Read existing course codes from the file
    with open('course_codes.txt', 'r') as f:
        course_codes = [line.strip() for line in f.readlines() if line.strip()]
        # Normalize input course code to lowercase
    
    # Check if the course code already exists
    if course_code in course_codes:
        messagebox.showerror("Error", f"Course code '{course_code}' already exists.")
        return

    # Add the new course code to the file
    with open('course_codes.txt', 'a') as f:
        f.write(course_code + "\n")
    messagebox.showinfo("Success", f"Course code '{course_code}' added successfully.")

#Function to delete a course
def delete_course_code(course_code):
    with open('course_codes.txt', 'r') as f:
        course_codes = [line.strip() for line in f.readlines() if line.strip()]

    if course_code in course_codes:
        course_codes.remove(course_code)
        with open('course_codes.txt', 'w') as f:
            for code in course_codes:
                f.write(code + "\n")
        messagebox.showinfo("Success", f"Course code '{course_code}' deleted successfully.")
    else:
        messagebox.showerror("Error", f"Course code '{course_code}' not found.")

# Create the Tkinter GUI
def main_window():
    root = tk.Tk()
    root.title("IT Department Class Attendance System(Admin's View)")
    root.geometry("800x500")
    root.resizable(False, False)

    title = tk.Label(root, text="CLASS ATTENDANCE SYSTEM (Admin's View)", font=("Helvetica", 23, "bold"))
    title.pack(pady=(20,20))

    def open_register():
        register_window = tk.Toplevel(root)
        register_window.title("Register")
        register_window.geometry("400x300")
        register_window.resizable(False, False)

        tk.Label(register_window, text="First Name:", font=("Helvetica", 12)).pack(pady=5)
        first_name_entry = tk.Entry(register_window, font=("Helvetica", 12))
        first_name_entry.pack(pady=5)

        tk.Label(register_window, text="Last Name:", font=("Helvetica", 12)).pack(pady=5)
        last_name_entry = tk.Entry(register_window, font=("Helvetica", 12))
        last_name_entry.pack(pady=5)

        tk.Label(register_window, text="Index Number:", font=("Helvetica", 12)).pack(pady=5)
        index_entry = tk.Entry(register_window, font=("Helvetica", 12))
        index_entry.pack(pady=5)

        def submit_registration():
            first_name = first_name_entry.get()
            last_name = last_name_entry.get()
            index_number = index_entry.get()
            if first_name and last_name and index_number:
                register_user(first_name, last_name, index_number)
                register_window.destroy()
            else:
                messagebox.showerror("Error", "All fields are required.")

        tk.Button(register_window, text="Register", font=("Helvetica", 12, "bold"), command=submit_registration).pack(pady=20)

    def open_attendance():
        mark_attendance()

    def open_options():
        options_window = tk.Toplevel(root)
        options_window.title("Options")
        options_window.geometry("400x300")
        options_window.resizable(False, False)

        tk.Button(options_window, text="Add Course Code", font=("Helvetica", 14), width=25, height=2, command=open_add_course_code).pack(pady=10)
        tk.Button(options_window, text="Delete Course Code", font=("Helvetica", 14), width=25, height=2, command=open_delete_course_code).pack(pady=10)
        

    def open_add_course_code():
        add_course_window = tk.Toplevel()
        add_course_window.title("Add Course Code")
        add_course_window.geometry("400x200")
        add_course_window.resizable(False, False)

        tk.Label(add_course_window, text="Enter Course Code:", font=("Helvetica", 12)).pack(pady=10)
        course_code_entry = tk.Entry(add_course_window, font=("Helvetica", 12))
        course_code_entry.pack(pady=10)

        def submit_course_code():
            course_code = course_code_entry.get().strip()
            if course_code:
                add_course_code(course_code)
                add_course_window.destroy()
            else:
                messagebox.showerror("Error", "Please enter a valid course code.")
        
        tk.Button(add_course_window, text="Submit", font=("Helvetica", 12, "bold"), command=submit_course_code).pack(pady=20)

    def open_delete_course_code():
        delete_course_window = tk.Toplevel()
        delete_course_window.title("Delete Course Code")
        delete_course_window.geometry("400x400")
        delete_course_window.resizable(False, False)

        tk.Label(delete_course_window, text="Select Course Code:", font=("Helvetica", 12)).pack(pady=10)

        course_codes = []
        with open('course_codes.txt', 'r') as f:
            course_codes = [line.strip() for line in f.readlines() if line.strip()]

        if not course_codes:
            messagebox.showerror("Error", "No course codes found.")
            delete_course_window.destroy()
            return
        
        frame = ttk.Frame(delete_course_window)
        frame.pack(fill='both', expand=True)

        tree = ttk.Treeview(frame)
        tree['columns'] = ('Course Code')
        tree.heading('#0', text='Course Code', anchor='w')
        

        for code in course_codes:
            tree.insert('', 'end', text=code)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(fill='both', expand=True)

        def delete_selected_course():
            selected_item = tree.selection()[0]
            course_code = tree.item(selected_item)['text']
            delete_course_code(course_code)
            tree.delete(selected_item)

        delete_button = tk.Button(delete_course_window, text="Delete Selected Course Code", command=delete_selected_course)
        delete_button.pack(pady=10)


    tk.Button(root, text="Register Student", font=("Helvetica", 14), command=open_register, width=25, height=2).pack(pady=5)
    tk.Button(root, text="Mark Attendance", font=("Helvetica", 14), command=open_attendance, width=25, height=2).pack(pady=5)
    tk.Button(root, text="Options", font=("Helvetica", 14), command=open_options, width=25, height=2).pack(pady=5)
    tk.Button(root, text="View All Students", font=("Helvetica", 14), command=open_view, width=25, height=2).pack(pady=5)
    tk.Button(root, text="Exit", font=("Helvetica", 14), command=root.quit, width=25, height=2).pack(pady=5)

    root.mainloop()

# Funcion to run the program
if __name__ == "__main__":
    main_window()