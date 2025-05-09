import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import pygame
import threading
import subprocess
import time
import os
import cv2

class VisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Psychedelic Music Visualizer")
        self.root.geometry("800x600")
        self.root.configure(bg="#1E1E2E")
        
        # Set app icon if available
        try:
            self.root.iconbitmap("icon.ico")  # You would need to create this icon file
        except:
            pass
            
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Track currently selected file
        self.selected_file = None
        self.visualizer_process = None
        self.preview_playing = False
        
        # Create main frame
        self.main_frame = tk.Frame(root, bg="#1E1E2E")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.create_header()
        
        # Create file selection area
        self.create_file_selection()
        
        # Create preview player
        self.create_preview_player()
        
        # Create visualizer settings
        self.create_visualizer_settings()
        
        # Create start visualizer button
        self.create_start_button()
        
        # Create footer
        self.create_footer()
        
        # Configure styling
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 12), background="#1E1E2E", foreground="white")
        self.style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'), background="#1E1E2E", foreground="#BB86FC")
        self.style.configure('Footer.TLabel', font=('Helvetica', 10), background="#1E1E2E", foreground="#6C7086")
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_header(self):
        header_frame = tk.Frame(self.main_frame, bg="#1E1E2E")
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(
            header_frame, 
            text="Psychedelic Music Visualizer", 
            style="Header.TLabel"
        )
        header_label.pack()
        
        subheader_label = ttk.Label(
            header_frame,
            text="Upload your music and create stunning visualizations",
            style="TLabel"
        )
        subheader_label.pack(pady=(5, 0))
        
    def create_file_selection(self):
        file_frame = tk.Frame(self.main_frame, bg="#1E1E2E")
        file_frame.pack(fill=tk.X, pady=10)
        
        file_label = ttk.Label(
            file_frame,
            text="Select Music File:",
            style="TLabel"
        )
        file_label.pack(anchor=tk.W)
        
        selection_frame = tk.Frame(file_frame, bg="#1E1E2E")
        selection_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.file_entry = ttk.Entry(selection_frame, width=50)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        browse_button = ttk.Button(
            selection_frame,
            text="Browse",
            command=self.browse_file
        )
        browse_button.pack(side=tk.RIGHT, padx=(10, 0))
        
    def create_preview_player(self):
        preview_frame = tk.Frame(self.main_frame, bg="#1E1E2E")
        preview_frame.pack(fill=tk.X, pady=20)
        
        preview_label = ttk.Label(
            preview_frame,
            text="Preview:",
            style="TLabel"
        )
        preview_label.pack(anchor=tk.W)
        
        controls_frame = tk.Frame(preview_frame, bg="#1E1E2E")
        controls_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.preview_button = ttk.Button(
            controls_frame,
            text="Play",
            command=self.toggle_preview
        )
        self.preview_button.pack(side=tk.LEFT)
        
        self.song_info = ttk.Label(
            controls_frame,
            text="No song selected",
            style="TLabel"
        )
        self.song_info.pack(side=tk.LEFT, padx=(10, 0))
        
    def create_visualizer_settings(self):
        settings_frame = tk.LabelFrame(self.main_frame, text="Visualizer Settings", bg="#1E1E2E", fg="white")
        settings_frame.pack(fill=tk.X, pady=20)
        
        # Resolution settings
        resolution_frame = tk.Frame(settings_frame, bg="#1E1E2E")
        resolution_frame.pack(fill=tk.X, padx=10, pady=10)
        
        resolution_label = ttk.Label(
            resolution_frame,
            text="Resolution:",
            style="TLabel"
        )
        resolution_label.pack(side=tk.LEFT)
        
        self.resolution_var = tk.StringVar(value="1280x720")
        resolution_options = ["800x600", "1024x768", "1280x720", "1920x1080", "Fullscreen"]
        resolution_dropdown = ttk.Combobox(
            resolution_frame,
            textvariable=self.resolution_var,
            values=resolution_options,
            state="readonly",
            width=15
        )
        resolution_dropdown.pack(side=tk.LEFT, padx=(10, 0))
        
        # Recording options
        recording_frame = tk.Frame(settings_frame, bg="#1E1E2E")
        recording_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.auto_record_var = tk.BooleanVar(value=False)
        auto_record_check = ttk.Checkbutton(
            recording_frame,
            text="Auto-record video",
            variable=self.auto_record_var
        )
        auto_record_check.pack(side=tk.LEFT)
        
        output_label = ttk.Label(
            recording_frame,
            text="Output folder:",
            style="TLabel"
        )
        output_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.output_var = tk.StringVar(value="recordings")
        output_entry = ttk.Entry(
            recording_frame,
            textvariable=self.output_var,
            width=20
        )
        output_entry.pack(side=tk.LEFT)
        
        # Effects settings
        effects_frame = tk.Frame(settings_frame, bg="#1E1E2E")
        effects_frame.pack(fill=tk.X, padx=10, pady=10)
        
        effects_label = ttk.Label(
            effects_frame,
            text="Enable Effects:",
            style="TLabel"
        )
        effects_label.grid(row=0, column=0, sticky=tk.W)
        
        # Effects checkboxes
        self.particles_var = tk.BooleanVar(value=True)
        particles_check = ttk.Checkbutton(
            effects_frame,
            text="Particles",
            variable=self.particles_var
        )
        particles_check.grid(row=0, column=1, padx=10)
        
        self.mandala_var = tk.BooleanVar(value=True)
        mandala_check = ttk.Checkbutton(
            effects_frame,
            text="Mandala",
            variable=self.mandala_var
        )
        mandala_check.grid(row=0, column=2, padx=10)
        
        self.ribbons_var = tk.BooleanVar(value=True)
        ribbons_check = ttk.Checkbutton(
            effects_frame,
            text="Ribbons",
            variable=self.ribbons_var
        )
        ribbons_check.grid(row=0, column=3, padx=10)
        
        self.fractals_var = tk.BooleanVar(value=True)
        fractals_check = ttk.Checkbutton(
            effects_frame,
            text="Fractals",
            variable=self.fractals_var
        )
        fractals_check.grid(row=1, column=1, padx=10)
        
        self.spectrum_var = tk.BooleanVar(value=True)
        spectrum_check = ttk.Checkbutton(
            effects_frame,
            text="Spectrum",
            variable=self.spectrum_var
        )
        spectrum_check.grid(row=1, column=2, padx=10)

        # Add Style settings
        style_frame = tk.Frame(settings_frame, bg="#1E1E2E")
        style_frame.pack(fill=tk.X, padx=10, pady=10)
        
        style_label = ttk.Label(
            style_frame,
            text="Visual Style:",
            style="TLabel"
        )
        style_label.pack(side=tk.LEFT)
        
        self.style_var = tk.StringVar(value="default")
        style_options = ["default", "cyberpunk", "industrial", "nature", "abstract"]
        style_dropdown = ttk.Combobox(
            style_frame,
            textvariable=self.style_var,
            values=style_options,
            state="readonly",
            width=15
        )
        style_dropdown.pack(side=tk.LEFT, padx=(10, 0))
        
        # Style strength slider
        strength_frame = tk.Frame(settings_frame, bg="#1E1E2E")
        strength_frame.pack(fill=tk.X, padx=10, pady=10)
        
        strength_label = ttk.Label(
            strength_frame,
            text="Style Strength:",
            style="TLabel"
        )
        strength_label.pack(side=tk.LEFT)
        
        self.strength_var = tk.DoubleVar(value=1.0)
        strength_slider = ttk.Scale(
            strength_frame,
            from_=0.1,
            to=1.0,
            variable=self.strength_var,
            orient=tk.HORIZONTAL,
            length=200
        )
        strength_slider.pack(side=tk.LEFT, padx=(10, 0))
        
        # Custom style training button
        train_button = ttk.Button(
            style_frame,
            text="Train Custom Style",
            command=self.open_style_trainer
        )
        train_button.pack(side=tk.RIGHT)
    
    # Add method to open style trainer
    def open_style_trainer(self):
        """Open the style trainer dialog"""
        trainer_window = tk.Toplevel(self.root)
        trainer_window.title("Train Custom Style")
        trainer_window.geometry("600x400")
        trainer_window.configure(bg="#1E1E2E")
        
        # Style name input
        name_frame = tk.Frame(trainer_window, bg="#1E1E2E")
        name_frame.pack(fill=tk.X, padx=20, pady=10)
        
        name_label = ttk.Label(
            name_frame,
            text="Style Name:",
            style="TLabel"
        )
        name_label.pack(side=tk.LEFT)
        
        name_entry = ttk.Entry(name_frame, width=30)
        name_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Image selection
        images_frame = tk.Frame(trainer_window, bg="#1E1E2E")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        images_label = ttk.Label(
            images_frame,
            text="Training Images:",
            style="TLabel"
        )
        images_label.pack(anchor=tk.W)
        
        images_listbox = tk.Listbox(images_frame, bg="#2E2E4E", fg="white", height=10)
        images_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        buttons_frame = tk.Frame(images_frame, bg="#1E1E2E")
        buttons_frame.pack(fill=tk.X)
        
        add_button = ttk.Button(
            buttons_frame,
            text="Add Images",
            command=lambda: self.add_training_images(images_listbox)
        )
        add_button.pack(side=tk.LEFT)
        
        clear_button = ttk.Button(
            buttons_frame,
            text="Clear List",
            command=lambda: images_listbox.delete(0, tk.END)
        )
        clear_button.pack(side=tk.LEFT, padx=(10, 0))
        
        # Training options
        options_frame = tk.Frame(trainer_window, bg="#1E1E2E")
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        epochs_label = ttk.Label(
            options_frame,
            text="Training Epochs:",
            style="TLabel"
        )
        epochs_label.grid(row=0, column=0, sticky=tk.W)
        
        epochs_var = tk.IntVar(value=20)
        epochs_spin = ttk.Spinbox(
            options_frame,
            from_=5,
            to=100,
            textvariable=epochs_var,
            width=5
        )
        epochs_spin.grid(row=0, column=1, padx=(10, 20))
        
        # Start training button
        train_button = tk.Button(
            trainer_window,
            text="Start Training",
            command=lambda: self.start_style_training(
                name_entry.get(),
                [images_listbox.get(i) for i in range(images_listbox.size())],
                epochs_var.get(),
                trainer_window
            ),
            bg="#BB86FC",
            fg="black",
            font=('Helvetica', 12, 'bold'),
            pady=8,
            borderwidth=0,
            cursor="hand2"
        )
        train_button.pack(fill=tk.X, padx=20, pady=20)
    
    def add_training_images(self, listbox):
        """Add training images to the listbox"""
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        )
        
        files = filedialog.askopenfilenames(
            title="Select Training Images",
            filetypes=filetypes
        )
        
        for file in files:
            listbox.insert(tk.END, file)

    def start_style_training(self, style_name, image_paths, epochs, window):
        """Start training a custom style"""
        if not style_name or len(image_paths) == 0:
            messagebox.showwarning("Training Error", "Style name and training images are required")
            return
        
        # Disable the training window during training
        window.title(f"Training '{style_name}' style... (this may take a while)")
        for child in window.winfo_children():
            if isinstance(child, (ttk.Button, tk.Button, ttk.Entry, tk.Listbox)):
                child.configure(state=tk.DISABLED)
        
        # Prepare training directory
        style_dir = f"training/styles/{style_name}"
        os.makedirs(style_dir, exist_ok=True)
        
        # Launch training in a separate thread
        thread = threading.Thread(
            target=self._run_training_process,
            args=(style_name, image_paths, epochs, window),
            daemon=True
        )
        thread.start()

    def _run_training_process(self, style_name, image_paths, epochs, window):
        """Run the training process in a background thread"""
        try:
            # Copy training images
            style_dir = f"training/styles/{style_name}"
            
            # Process images
            for i, path in enumerate(image_paths):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (256, 256))
                    output_path = os.path.join(style_dir, f"train_{i:04d}.jpg")
                    cv2.imwrite(output_path, img)
            
            # Prepare training command
            cmd = [
                sys.executable,
                "train_style_models.py",
                "--style", style_name,
                "--epochs", str(epochs)
            ]
            
            # Run training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Read output
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                print(line.strip())
            
            # Re-enable window and show success message
            self.root.after(
                0,
                lambda: [
                    window.destroy(),
                    messagebox.showinfo(
                        "Training Complete", 
                        f"The '{style_name}' style has been trained successfully!"
                    )
                ]
            )
        except Exception as e:
            # Show error message
            self.root.after(
                0,
                lambda: [
                    window.destroy(),
                    messagebox.showerror(
                        "Training Error", 
                        f"An error occurred during training: {str(e)}"
                    )
                ]
            )
        
    def create_start_button(self):
        button_frame = tk.Frame(self.main_frame, bg="#1E1E2E")
        button_frame.pack(fill=tk.X, pady=20)
        
        start_button = tk.Button(
            button_frame,
            text="Start Visualizer",
            command=self.launch_visualizer,
            bg="#BB86FC",
            fg="black",
            font=('Helvetica', 14, 'bold'),
            pady=10,
            borderwidth=0,
            cursor="hand2"
        )
        start_button.pack(fill=tk.X)
        
        # Add hover effect
        start_button.bind("<Enter>", lambda e: e.widget.config(bg="#A66BFC"))
        start_button.bind("<Leave>", lambda e: e.widget.config(bg="#BB86FC"))
        
    def create_footer(self):
        footer_frame = tk.Frame(self.main_frame, bg="#1E1E2E")
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        footer_text = "ESC: exit • Keys 1-5: toggle effects • C: change colors • R: toggle recording • Space: pause/play"
        footer_label = ttk.Label(
            footer_frame,
            text=footer_text,
            style="Footer.TLabel",
            justify=tk.CENTER
        )
        footer_label.pack()
        
    def browse_file(self):
        filetypes = (
            ("Audio files", "*.mp3 *.wav *.ogg"),
            ("All files", "*.*")
        )
        
        file_path = filedialog.askopenfilename(
            title="Select a music file",
            filetypes=filetypes
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, file_path)
            
            # Update song info
            filename = os.path.basename(file_path)
            self.song_info.config(text=filename)
            
            # Stop any playing preview
            if self.preview_playing:
                self.toggle_preview()
    
    def toggle_preview(self):
        if not self.selected_file:
            messagebox.showinfo("Info", "Please select a music file first")
            return
            
        if not self.preview_playing:
            # Start preview
            try:
                pygame.mixer.music.load(self.selected_file)
                pygame.mixer.music.play()
                self.preview_playing = True
                self.preview_button.config(text="Stop")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play file: {str(e)}")
        else:
            # Stop preview
            pygame.mixer.music.stop()
            self.preview_playing = False
            self.preview_button.config(text="Play")
    
    def launch_visualizer(self):
        if not self.selected_file:
            messagebox.showinfo("Info", "Please select a music file first")
            return
            
        # Stop any playing preview
        if self.preview_playing:
            self.toggle_preview()
            
        # Get resolution
        resolution = self.resolution_var.get()
        width, height = 1280, 720  # Default
        fullscreen = False
        
        if resolution != "Fullscreen":
            width, height = map(int, resolution.split("x"))
        else:
            fullscreen = True
        
        # Create output directory if it doesn't exist
        output_dir = self.output_var.get()
        if self.auto_record_var.get() and output_dir:
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except Exception as e:
                    messagebox.showwarning("Warning", f"Could not create output directory: {str(e)}")
            
            # Create output filename
            filename = os.path.basename(self.selected_file)
            base_name, _ = os.path.splitext(filename)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{base_name}_{timestamp}.mp4")
        else:
            output_file = None
            
        # Construct command-line arguments
        args = [
            sys.executable,
            "main_visualizer.py",
            self.selected_file,
            "--width", str(width),
            "--height", str(height)
        ]
        
        args.extend(["--style", self.style_var.get()])
        args.extend(["--style-strength", str(self.strength_var.get())])
        
        if fullscreen:
            args.append("--fullscreen")
            
        # Add effect settings
        if not self.particles_var.get():
            args.append("--no-particles")
        if not self.mandala_var.get():
            args.append("--no-mandala")
        if not self.ribbons_var.get():
            args.append("--no-ribbons")
        if not self.fractals_var.get():
            args.append("--no-fractals")
        if not self.spectrum_var.get():
            args.append("--no-spectrum")
            
        # Add recording options
        if self.auto_record_var.get():
            args.append("--record")
            if output_file:
                args.extend(["--output", output_file])
                
        # Launch visualizer in a separate process
        try:
            self.visualizer_process = subprocess.Popen(args)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch visualizer: {str(e)}")
    
    def on_closing(self):
        # Stop any playing preview
        if self.preview_playing:
            pygame.mixer.music.stop()
            
        # Terminate visualizer process if running
        if self.visualizer_process and self.visualizer_process.poll() is None:
            self.visualizer_process.terminate()
            
        self.root.destroy()