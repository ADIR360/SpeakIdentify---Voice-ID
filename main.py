import tkinter as tk
from tkinter import messagebox, simpledialog
import audio_processing as ap
import database


class VoiceIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Identification Tool")

        # Create database and load patterns
        database.create_database()
        self.voice_db = database.load_database()

        # Create Record Button
        self.record_button = tk.Button(root, text="Record Voice to Database", command=self.open_record_window)
        self.record_button.pack(pady=10)

        # Create Analyze Button
        self.analyze_button = tk.Button(root, text="Analyze Voice", command=self.open_analyze_window)
        self.analyze_button.pack(pady=10)

    def open_record_window(self):
        self.record_window = tk.Toplevel(self.root)
        self.record_window.title("Record Voice")

        # Prompt for name
        tk.Label(self.record_window, text="Enter the letter or name for this voice pattern:").pack(pady=10)
        self.name_entry = tk.Entry(self.record_window)
        self.name_entry.pack(pady=10)

        # Start recording button
        self.start_record_button = tk.Button(self.record_window, text="Start Recording", command=self.record_voice)
        self.start_record_button.pack(pady=10)

    def record_voice(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Please enter a name or letter.")
            return

        messagebox.showinfo("Recording", "Start speaking now...")  # Feedback
        recorded_audio = ap.record_audio(duration=3)  # Record audio for 3 seconds
        processed_audio = ap.process_audio(recorded_audio)  # Process the recorded audio

        # Save voice pattern to the database
        database.save_voice_pattern(name, processed_audio)
        self.voice_db[name] = processed_audio  # Update in-memory database
        messagebox.showinfo("Success", f"Voice pattern for '{name}' recorded successfully!")
        self.record_window.destroy()  # Close record window after recording

    def open_analyze_window(self):
        self.analyze_window = tk.Toplevel(self.root)
        self.analyze_window.title("Analyze Voice")

        tk.Label(self.analyze_window, text="Press 'Start Analyzing' and speak").pack(pady=10)

        # Start analyzing button
        self.start_analyze_button = tk.Button(self.analyze_window, text="Start Analyzing", command=self.analyze_voice)
        self.start_analyze_button.pack(pady=10)

    def analyze_voice(self):
        messagebox.showinfo("Analyzing", "Start speaking now...")  # Feedback
        recorded_audio = ap.record_audio(duration=3)  # Record audio for 3 seconds
        processed_audio = ap.process_audio(recorded_audio)  # Process the recorded audio

        # Identify the speaker based on the processed audio
        identified_speaker = ap.identify_speaker(processed_audio, self.voice_db)

        if identified_speaker:
            messagebox.showinfo("Result", f"Identified Speaker: {identified_speaker}")
        else:
            messagebox.showinfo("Result", "Speaker not recognized.")
        self.analyze_window.destroy()  # Close analyze window after analysis


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceIDApp(root)
    root.mainloop()
