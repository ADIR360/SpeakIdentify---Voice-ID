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
from tkinter import ttk
import collections
import threading
import time
import sounddevice as sd
import database
import os
import glob


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

        self.fig = Figure(figsize=(9, 2.6), dpi=100)
        self.ax_pitch = self.fig.add_subplot(131)
        self.ax_pitch.set_title('Pitch (F0)')
        self.ax_pitch.set_xlabel('Time (s)')
        self.ax_pitch.set_ylabel('Hz')
        self.ax_pitch.set_ylim(50, 500)
        self.ax_spec = self.fig.add_subplot(132)
        self.ax_spec.set_title('Mel Spectrogram')
        self.ax_spec.set_xlabel('Time')
        self.ax_spec.set_ylabel('Mel bin')
        self.ax_match = self.fig.add_subplot(133)
        self.ax_match.set_title('Top Matches')

        self.canvas = FigureCanvasTkAgg(self.fig, master=vis)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Hover annotation for charts
        self._hover_annot = self.ax_match.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#fffffe", ec="#333333", alpha=0.9),
            arrowprops=dict(arrowstyle="->")
        )
        self._hover_annot.set_visible(False)
        self._bars = []  # bar artists for match plot
        self.canvas.mpl_connect('motion_notify_event', self._on_motion_hover)

        # Menu for importing audio files (public figure voices)
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import Voice (WAV/MP3)", command=self.import_voice_file)
        filemenu.add_command(label="Import Dataset Folder", command=self.import_dataset_folder)
        manage_menu = tk.Menu(menubar, tearoff=0)
        manage_menu.add_command(label="List Users", command=self.show_user_list)
        manage_menu.add_command(label="Delete User", command=self.delete_user_prompt)
        manage_menu.add_command(label="Rename User", command=self.rename_user_prompt)
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Voice Emulator", command=self.open_voice_emulator)
        menubar.add_cascade(label="File", menu=filemenu)
        menubar.add_cascade(label="Manage", menu=manage_menu)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        self.root.config(menu=menubar)

        # Live monitor controls
        live = tk.LabelFrame(root, text="Live Monitor", padx=8, pady=8)
        live.pack(fill=tk.X, expand=False, padx=12, pady=(0, 12))
        tk.Button(live, text="Start Live Monitor", command=self.start_live_monitor).pack(side=tk.LEFT)
        tk.Button(live, text="Stop Live Monitor", command=self.stop_live_monitor).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(live, text="VU:").pack(side=tk.LEFT, padx=(12, 4))
        self.vu_var = tk.DoubleVar(value=0.0)
        self.vu_bar = ttk.Progressbar(live, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=100.0, variable=self.vu_var)
        self.vu_bar.pack(side=tk.LEFT)

        # Live stream state
        self._stream = None
        self._stream_buffer = collections.deque(maxlen=ap.DEFAULT_SAMPLE_RATE * 3)  # last 3s
        self._stream_lock = threading.Lock()
        self._live_after_id = None
        self._running_live = False

        # Simple Tk tooltips for key numeric controls
        self._tooltip = _Tooltip(self.root)
        self._tooltip.attach(self.threshold_scale, "Decision threshold for cosine similarity. Higher = stricter match.")
        self._tooltip.attach(self.vu_bar, "Live input level (VU). Aim for mid-range for best results.")

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
            # Update per-user acoustic stats
            mean_f0, mfcc_mean = ap.compute_clip_stats(recorded_audio)
            database.update_user_stats(name, mean_f0, mfcc_mean)

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
            self.update_spectrogram_plot(recorded_audio)

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
            bars = self.ax_match.barh(y, scores, color='#2e7d32', picker=False)
            self._bars = list(bars)
            self.ax_match.set_yticks(y)
            self.ax_match.set_yticklabels(labels)
            self.ax_match.set_xlim(0.0, 1.0)
            self.ax_match.invert_yaxis()
            self.canvas.draw_idle()
        except Exception as e:
            self.status_label.config(text=f"Match plot error: {e}", fg="#c62828")

    def update_spectrogram_plot(self, audio_array: np.ndarray):
        try:
            S_db = ap.compute_spectrogram(audio_array)
            self.ax_spec.clear()
            self.ax_spec.set_title('Mel Spectrogram')
            self.ax_spec.set_xlabel('Time')
            self.ax_spec.set_ylabel('Mel bin')
            self.ax_spec.imshow(S_db, origin='lower', aspect='auto', interpolation='nearest')
            self.canvas.draw_idle()
        except Exception as e:
            self.status_label.config(text=f"Spectrogram error: {e}", fg="#c62828")

    def _on_motion_hover(self, event):
        # Hover handler for pitch, spectrogram, and match bars
        vis = False
        if event.inaxes is self.ax_match and self._bars:
            for idx, bar in enumerate(self._bars):
                if bar.contains(event)[0]:
                    width = float(bar.get_width())
                    label = self.ax_match.get_yticklabels()[idx].get_text()
                    text = f"{label}: {width:.2f}\nCosine similarity (1.0 = identical)."
                    self._set_hover_annot((event.xdata, event.ydata), text)
                    vis = True
                    break
        elif event.inaxes is self.ax_pitch and event.xdata is not None and event.ydata is not None:
            t = max(0.0, float(event.xdata))
            f0 = max(0.0, float(event.ydata))
            text = f"t={t:.2f}s, F0={f0:.1f} Hz\nFundamental frequency (pitch)."
            self._set_hover_annot((event.xdata, event.ydata), text)
            vis = True
        elif event.inaxes is self.ax_spec and event.xdata is not None and event.ydata is not None:
            text = "Mel spectrogram\nBrightness ~ energy."
            self._set_hover_annot((event.xdata, event.ydata), text)
            vis = True

        if not vis and self._hover_annot.get_visible():
            self._hover_annot.set_visible(False)
            self.canvas.draw_idle()

    def _set_hover_annot(self, xy, text):
        self._hover_annot.xy = xy
        self._hover_annot.set_text(text)
        if not self._hover_annot.get_visible():
            self._hover_annot.set_visible(True)
        self.canvas.draw_idle()

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
            # Update stats
            mean_f0, mfcc_mean = ap.compute_clip_stats(audio, sample_rate=ap.DEFAULT_SAMPLE_RATE)
            database.update_user_stats(name, mean_f0, mfcc_mean)
            self.voice_db = database.load_database()
            self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}")
            self.status_label.config(text=f"Imported '{name}' from file", fg="#2e7d32")
        except Exception as e:
            self.status_label.config(text=f"Import error: {e}", fg="#c62828")

    def import_dataset_folder(self):
        folder = filedialog.askdirectory(title='Select dataset folder (subfolders per person)')
        if not folder:
            return
        self.status_label.config(text="Importing dataset...", fg="#1565c0")

        def worker():
            count_files = 0
            persons_added = set()
            try:
                # Expect structure: root/person_a/*.wav, root/person_b/*.mp3, ...
                # If the chosen folder itself has audio files, prompt for one name
                audio_ext = ("*.wav", "*.mp3", "*.flac", "*.m4a")
                subdirs = [d for d in sorted(os.listdir(folder)) if os.path.isdir(os.path.join(folder, d))]
                if not subdirs:
                    files = []
                    for pat in audio_ext:
                        files.extend(glob.glob(os.path.join(folder, pat)))
                    if files:
                        name = simpledialog.askstring('Label', 'Enter name/label for this dataset:')
                        if not name:
                            raise RuntimeError('No label provided for flat folder dataset.')
                        for f in files:
                            audio, _ = librosa.load(f, sr=ap.DEFAULT_SAMPLE_RATE, mono=True)
                            emb = ap.process_audio(audio)
                            database.save_voice_pattern(name, emb)
                            mean_f0, mfcc_mean = ap.compute_clip_stats(audio)
                            database.update_user_stats(name, mean_f0, mfcc_mean)
                            count_files += 1
                        persons_added.add(name)
                else:
                    for person in subdirs:
                        person_path = os.path.join(folder, person)
                        files = []
                        for pat in audio_ext:
                            files.extend(glob.glob(os.path.join(person_path, pat)))
                        if not files:
                            continue
                        for f in files:
                            audio, _ = librosa.load(f, sr=ap.DEFAULT_SAMPLE_RATE, mono=True)
                            emb = ap.process_audio(audio)
                            database.save_voice_pattern(person, emb)
                            mean_f0, mfcc_mean = ap.compute_clip_stats(audio)
                            database.update_user_stats(person, mean_f0, mfcc_mean)
                            count_files += 1
                        persons_added.add(person)
                # Reload DB at the end
                self.voice_db = database.load_database()
                self.root.after(0, lambda: self.users_label.config(text=f"Enrolled users: {len(self.voice_db)}"))
                self.root.after(0, lambda: self.status_label.config(text=f"Imported {count_files} files for {len(persons_added)} users", fg="#2e7d32"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"Dataset import error: {e}", fg="#c62828"))

        threading.Thread(target=worker, daemon=True).start()

    def open_voice_emulator(self):
        win = tk.Toplevel(self.root)
        win.title('Voice Emulator')
        tk.Label(win, text='Select target voice (from imported dataset):').pack(pady=(8, 4))
        users = database.list_users()
        if not users:
            tk.Label(win, text='No users available. Import a dataset first.').pack(pady=8)
            return
        target_var = tk.StringVar(value=users[0])
        combo = ttk.Combobox(win, values=users, textvariable=target_var, state='readonly')
        combo.pack(pady=4, fill=tk.X, padx=12)

        tk.Button(win, text='Record and Convert (3s)', command=lambda: self._emulate_now(win, target_var.get())).pack(pady=10)

    def _emulate_now(self, parent, target_name: str):
        try:
            messagebox.showinfo('Voice Emulator', 'Recording 3 seconds. Speak now...')
            src_audio = ap.record_audio(duration=3)
            self.update_pitch_plot(src_audio)
            self.update_spectrogram_plot(src_audio)
            # Source stats from recording
            src_stats = ap.compute_clip_stats(src_audio)
            # Target stats from DB
            tgt_stats = database.get_user_stats(target_name)
            if tgt_stats is None:
                # Fallback: estimate from the same audio to avoid extreme changes
                tgt_stats = src_stats
            converted = ap.voice_convert(src_audio, src_stats, tgt_stats)
            # Playback converted audio
            sd.play(converted, samplerate=ap.DEFAULT_SAMPLE_RATE, blocking=False)
            messagebox.showinfo('Voice Emulator', f'Playing converted audio towards "{target_name}"')
        except Exception as e:
            messagebox.showerror('Voice Emulator Error', str(e))

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

    # ===== Live streaming monitor =====
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            # Non-fatal; could log status
            pass
        data = indata[:, 0].copy()
        with self._stream_lock:
            self._stream_buffer.extend(data)
        # Update VU based on RMS of this block
        rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
        # Map to 0..100 (simple linear mapping)
        vu = max(0.0, min(100.0, rms * 4000.0))
        # Tk variables should be set in main thread; schedule via after
        self.root.after(0, lambda: self.vu_var.set(vu))

    def start_live_monitor(self):
        if self._running_live:
            return
        try:
            self._stream_buffer.clear()
            self._running_live = True
            self._stream = sd.InputStream(
                samplerate=ap.DEFAULT_SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=256,
                callback=self._audio_callback,
            )
            self._stream.start()
            # Kick off periodic plot updates
            self._schedule_live_update()
            self.status_label.config(text="Live monitor: running", fg="#2e7d32")
        except Exception as e:
            self._running_live = False
            self.status_label.config(text=f"Live monitor error: {e}", fg="#c62828")

    def stop_live_monitor(self):
        self._running_live = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        self.status_label.config(text="Live monitor: stopped", fg="#ef6c00")

    def _schedule_live_update(self):
        # Update pitch plot every 100 ms using last ~2s of audio
        if not self._running_live:
            return
        try:
            with self._stream_lock:
                if len(self._stream_buffer) == 0:
                    data = None
                else:
                    buf = np.array(self._stream_buffer, dtype=np.float32)
                    # take last 2 seconds for pitch
                    n_last = ap.DEFAULT_SAMPLE_RATE * 2
                    data = buf[-n_last:]
            if data is not None and data.size > 256:
                self.update_pitch_plot(data)
                self.update_spectrogram_plot(data)
        except Exception as e:
            self.status_label.config(text=f"Live update error: {e}", fg="#c62828")
        finally:
            self._live_after_id = self.root.after(100, self._schedule_live_update)


class _Tooltip:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.tip = None
        self.after_id = None

    def attach(self, widget, text: str):
        def on_enter(_):
            self._show(widget, text)
        def on_leave(_):
            self._hide()
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)

    def _show(self, widget, text: str):
        if self.tip is not None:
            self._hide()
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.root)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tip, text=text, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, font=("tahoma", "9", "normal"))
        label.pack(ipadx=4, ipady=2)
        # Auto-hide after a while
        self.after_id = self.root.after(4000, self._hide)

    def _hide(self):
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.tip is not None:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceIDApp(root)
    root.mainloop()
