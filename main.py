import tkinter as tk
from tkinter import messagebox, simpledialog
import audio_processing as ap
import numpy as np
import librosa
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog
import database


class VoiceIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Identification Tool")

        # Create database and load patterns
        database.create_database()
        self.voice_db = database.load_database()

        # UI State
        self.threshold_var = tk.DoubleVar(value=0.6)

        # Top container
        container = tk.Frame(root, padx=12, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        # Threshold control
        tk.Label(container, text="Similarity Threshold (0.10 - 0.95)").grid(row=0, column=0, sticky="w")
        self.threshold_scale = tk.Scale(
            container,
            from_=0.10,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            length=260,
        )
        self.threshold_scale.grid(row=0, column=1, sticky="we", padx=(8, 0))

        # Threshold presets and auto-calibrate
        presets = tk.Frame(container)
        presets.grid(row=3, column=0, columnspan=2, sticky="we", pady=(6, 0))
        tk.Button(presets, text="Lenient", command=lambda: self.threshold_var.set(0.45)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(presets, text="Balanced", command=lambda: self.threshold_var.set(0.60)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(presets, text="Strict", command=lambda: self.threshold_var.set(0.75)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(presets, text="Auto-Calibrate", command=self.auto_calibrate_threshold).pack(side=tk.LEFT)

        # Record / Analyze buttons
        self.record_button = tk.Button(container, text="Record Voice to Database", command=self.open_record_window)
        self.record_button.grid(row=1, column=0, pady=(12, 8), sticky="we")

        self.analyze_button = tk.Button(container, text="Analyze Voice", command=self.open_analyze_window)
        self.analyze_button.grid(row=1, column=1, pady=(12, 8), sticky="we", padx=(8, 0))

        # Info row
        self.users_label = tk.Label(container, text=f"Enrolled users: {len(self.voice_db)}")
        self.users_label.grid(row=2, column=0, sticky="w", pady=(6, 0))

        self.status_label = tk.Label(container, text="Status: Ready", fg="#2e7d32")
        self.status_label.grid(row=2, column=1, sticky="e", pady=(6, 0))

        # Make columns responsive
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # Visualization area
        vis = tk.LabelFrame(root, text="Visualization", padx=8, pady=8)
        vis.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.fig = Figure(figsize=(6, 2.2), dpi=100)
        self.ax_pitch = self.fig.add_subplot(121)
        self.ax_pitch.set_title('Pitch (F0)')
        self.ax_pitch.set_xlabel('Time (s)')
        self.ax_pitch.set_ylabel('Hz')
        self.ax_pitch.set_ylim(50, 500)
        self.ax_match = self.fig.add_subplot(122)
        self.ax_match.set_title('Top Matches')

        self.canvas = FigureCanvasTkAgg(self.fig, master=vis)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Menu for importing audio files (public figure voices)
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import Voice (WAV/MP3)", command=self.import_voice_file)
        manage_menu = tk.Menu(menubar, tearoff=0)
        manage_menu.add_command(label="List Users", command=self.show_user_list)
        manage_menu.add_command(label="Delete User", command=self.delete_user_prompt)
        manage_menu.add_command(label="Rename User", command=self.rename_user_prompt)
        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Manage", menu=manage_menu)
        self.root.config(menu=menubar)

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

        try:
            messagebox.showinfo("Recording", "Start speaking now...")
            recorded_audio = ap.record_audio(duration=3)
            processed_audio = ap.process_audio(recorded_audio)

            # Save voice pattern to the database (also appends to samples)
            database.save_voice_pattern(name, processed_audio)

            # Refresh in-memory DB to use centroids
            self.voice_db = database.load_database()
            self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}")
            self.status_label.config(text=f"Status: Enrolled '{name}'", fg="#2e7d32")

            # Update pitch visualization from the recorded audio
            self.update_pitch_plot(recorded_audio)
            messagebox.showinfo("Success", f"Voice sample for '{name}' recorded successfully!")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", fg="#c62828")
            messagebox.showerror("Recording Error", str(e))
        finally:
            self.record_window.destroy()  # Close record window after recording

    def open_analyze_window(self):
        self.analyze_window = tk.Toplevel(self.root)
        self.analyze_window.title("Analyze Voice")

        tk.Label(self.analyze_window, text="Press 'Start Analyzing' and speak").pack(pady=10)

        # Start analyzing button
        self.start_analyze_button = tk.Button(self.analyze_window, text="Start Analyzing", command=self.analyze_voice)
        self.start_analyze_button.pack(pady=10)

    def analyze_voice(self):
        try:
            messagebox.showinfo("Analyzing", "Start speaking now...")
            recorded_audio = ap.record_audio(duration=3)
            processed_audio = ap.process_audio(recorded_audio)
            # Update pitch visualization from the analyzed audio
            self.update_pitch_plot(recorded_audio)

            # Identify the speaker based on the processed audio using UI threshold
            threshold = float(self.threshold_var.get())
            identified_speaker = ap.identify_speaker(processed_audio, self.voice_db, similarity_threshold=threshold)

            # Show top-5 matches bar chart
            top = ap.rank_similarities(processed_audio, self.voice_db, top_k=5)
            self.update_match_plot(top)

            # Heuristic AI-detection score
            ai_score = ap.estimate_ai_score(recorded_audio)
            ai_pct = int(round(ai_score * 100))

            if identified_speaker:
                self.status_label.config(text=f"Result: {identified_speaker} | AI-likelihood: {ai_pct}%", fg="#2e7d32")
                messagebox.showinfo("Result", f"Identified Speaker: {identified_speaker}\nAI-likelihood: {ai_pct}%")
            else:
                self.status_label.config(text=f"Result: Not recognized | AI-likelihood: {ai_pct}%", fg="#ef6c00")
                messagebox.showinfo("Result", f"Speaker not recognized.\nAI-likelihood: {ai_pct}%")
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", fg="#c62828")
            messagebox.showerror("Analyze Error", str(e))
        finally:
            self.analyze_window.destroy()  # Close analyze window after analysis

    def show_user_list(self):
        try:
            users = database.list_users()
            if not users:
                messagebox.showinfo('Users', 'No enrolled users found.')
                return
            messagebox.showinfo('Users', '\n'.join(users))
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def delete_user_prompt(self):
        try:
            name = simpledialog.askstring('Delete User', 'Enter user name to delete:')
            if not name:
                return
            database.delete_user(name)
            self.voice_db = database.load_database()
            self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}")
            messagebox.showinfo('Deleted', f"Deleted '{name}'.")
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def rename_user_prompt(self):
        try:
            old = simpledialog.askstring('Rename User', 'Enter current user name:')
            if not old:
                return
            new = simpledialog.askstring('Rename User', 'Enter new user name:')
            if not new:
                return
            database.rename_user(old, new)
            self.voice_db = database.load_database()
            self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}")
            messagebox.showinfo('Renamed', f"Renamed '{old}' to '{new}'.")
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def update_pitch_plot(self, audio_array: np.ndarray):
        try:
            t, f0 = ap.compute_pitch_series(audio_array)
            self.ax_pitch.clear()
            self.ax_pitch.set_title('Pitch (F0)')
            self.ax_pitch.set_xlabel('Time (s)')
            self.ax_pitch.set_ylabel('Hz')
            self.ax_pitch.set_ylim(50, 500)
            self.ax_pitch.plot(t, f0, color='#1565c0')
            self.canvas.draw_idle()
        except Exception as e:
            self.status_label.config(text=f"Pitch plot error: {e}", fg="#c62828")

    def update_match_plot(self, top_scores):
        try:
            self.ax_match.clear()
            self.ax_match.set_title('Top Matches')
            if not top_scores:
                self.ax_match.text(0.5, 0.5, 'No data', ha='center', va='center')
                self.canvas.draw_idle()
                return
            labels, scores = zip(*top_scores)
            y = np.arange(len(labels))
            self.ax_match.barh(y, scores, color='#2e7d32')
            self.ax_match.set_yticks(y)
            self.ax_match.set_yticklabels(labels)
            self.ax_match.set_xlim(0.0, 1.0)
            self.ax_match.invert_yaxis()
            self.canvas.draw_idle()
        except Exception as e:
            self.status_label.config(text=f"Match plot error: {e}", fg="#c62828")

    def import_voice_file(self):
        try:
            file_path = filedialog.askopenfilename(
                title='Select audio file',
                filetypes=[('Audio Files', '*.wav *.mp3 *.flac *.m4a'), ('All Files', '*.*')]
            )
            if not file_path:
                return

            # Ask for name label
            name = simpledialog.askstring('Label', 'Enter name/label for this voice:')
            if not name:
                return

            # Load audio file, resample to 16kHz mono
            audio, sr = librosa.load(file_path, sr=ap.DEFAULT_SAMPLE_RATE, mono=True)
            emb = ap.process_audio(audio, sample_rate=ap.DEFAULT_SAMPLE_RATE)
            database.save_voice_pattern(name, emb)
            self.voice_db = database.load_database()
            self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}")
            self.status_label.config(text=f"Imported '{name}' from file", fg="#2e7d32")
        except Exception as e:
            self.status_label.config(text=f"Import error: {e}", fg="#c62828")

    def auto_calibrate_threshold(self):
        """Estimate a reasonable threshold from current enrolled users by computing pairwise similarities between different users' centroids.

        Sets threshold slightly above the maximum impostor similarity.
        """
        try:
            names = list(self.voice_db.keys())
            if len(names) < 2:
                messagebox.showinfo('Auto-Calibrate', 'Need at least 2 users to auto-calibrate.')
                return
            # Compute pairwise impostor similarities
            centroids = [self.voice_db[n] for n in names]
            def norm(v):
                v = np.asarray(v, dtype=np.float32)
                nz = np.linalg.norm(v) + 1e-8
                return v / nz
            normed = [norm(v) for v in centroids]
            max_impostor = -1.0
            for i in range(len(normed)):
                for j in range(i+1, len(normed)):
                    s = float(np.dot(normed[i], normed[j]))
                    if s > max_impostor:
                        max_impostor = s
            # Set threshold a small margin above max impostor
            new_t = float(min(0.95, max(0.10, max_impostor + 0.05)))
            self.threshold_var.set(new_t)
            self.status_label.config(text=f"Auto-calibrated threshold: {new_t:.2f}", fg="#2e7d32")
        except Exception as e:
            self.status_label.config(text=f"Calibration error: {e}", fg="#c62828")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceIDApp(root)
    root.mainloop()
