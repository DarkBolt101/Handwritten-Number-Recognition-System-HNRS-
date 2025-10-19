"""
Main GUI Application for Digit Recognition System
"""

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw
import threading

from segmentation import segment_image_combined
from digit_extraction import extract_digits_combined, create_chips_display, get_last_segmentation_method
from splitting import get_last_splitting_method
from models import ModelManager

# =============================================================================
# MAIN GUI CLASS
# =============================================================================

class DigitRecognitionGUI(tk.Tk):
    """Main GUI for digit recognition."""
    
    def __init__(self):
        super().__init__()
        self._init_variables()
        self._setup_ui()
        self._setup_callbacks()
        self._load_models()
        self._start_auto_update()
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def _init_variables(self):
        """Initialize GUI variables."""
        self.title("Digit Recognition System")
        self.geometry("1400x900")
        self.configure(bg='#f0f0f0')
        
        # Model manager
        self.model_manager = ModelManager()
        
        # Image variables
        self.draw_img = Image.new("L", (600, 500), color=255)
        self.draw = ImageDraw.Draw(self.draw_img)
        self.current_gray = None
        self.current_mask = None
        self.digit_chips = []
        self.last_xy = None
        
        # GUI variables
        self.method_var = tk.StringVar(value="auto")
        self.model_var = tk.StringVar(value="auto")
        self.confidence_threshold = tk.DoubleVar(value=0.1)
        self.foreground_threshold = tk.DoubleVar(value=0.05)
        self.split_method_var = tk.StringVar(value="auto")
        self.use_grayscale_chips = tk.BooleanVar(value=True)
        self.enable_pre_erosion = tk.BooleanVar(value=False)
        
        # Processing state
        self.is_processing = False
        self.processing_thread = None
        
        # Statistics
        self.total_processed = 0
        self.successful_detections = 0
        self.last_model_used = "Unknown"
        self.last_segmentation_used = "Unknown"
        self.last_splitting_method_used = "None"
    
    def _setup_ui(self):
        """Set up the user interface."""
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self._create_left_panel(main_frame)
        self._create_right_panel(main_frame)
    
    def _create_left_panel(self, parent):
        """Create left control panel."""
        left_frame = ttk.Frame(parent)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self._create_controls_section(left_frame)
        self._create_settings_section(left_frame)
        self._create_advanced_section(left_frame)
        self._create_canvas_section(left_frame)
    
    def _create_controls_section(self, parent):
        """Create controls section."""
        controls_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Open Image", command=self._open_image).grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Process", command=self._start_processing).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Clear", command=self._clear_canvas).grid(row=0, column=2, padx=(5, 0), sticky=(tk.W, tk.E))
        
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
    
    def _create_settings_section(self, parent):
        """Create settings section."""
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Model selection
        ttk.Label(settings_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        model_combo = ttk.Combobox(settings_frame, textvariable=self.model_var, 
                                  values=["auto", "cnn", "svm", "rf"], state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Segmentation method
        ttk.Label(settings_frame, text="Segmentation:").grid(row=1, column=0, sticky=tk.W, pady=2)
        seg_combo = ttk.Combobox(settings_frame, textvariable=self.method_var,
                                values=["auto", "otsu", "adaptive", "kmeans", "local_threshold", "canny_edges"],
                                state="readonly", width=15)
        seg_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Split method
        ttk.Label(settings_frame, text="Split Method:").grid(row=2, column=0, sticky=tk.W, pady=2)
        split_combo = ttk.Combobox(settings_frame, textvariable=self.split_method_var,
                                  values=["auto", "simple", "projection", "kmeans1d", "skeleton"],
                                  state="readonly", width=15)
        split_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=3, column=0, sticky=tk.W, pady=2)
        conf_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.confidence_threshold,
                              orient=tk.HORIZONTAL, length=150)
        conf_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        self.conf_label = ttk.Label(settings_frame, text=f"{self.confidence_threshold.get():.2f}")
        self.conf_label.grid(row=3, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        conf_scale.configure(command=self._update_conf_label)
        
        # Foreground threshold
        ttk.Label(settings_frame, text="Foreground Threshold:").grid(row=4, column=0, sticky=tk.W, pady=2)
        fg_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.foreground_threshold,
                            orient=tk.HORIZONTAL, length=150)
        fg_scale.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=2, padx=(5, 0))
        self.fg_label = ttk.Label(settings_frame, text=f"{self.foreground_threshold.get():.2f}")
        self.fg_label.grid(row=4, column=2, sticky=tk.W, pady=2, padx=(5, 0))
        fg_scale.configure(command=self._update_fg_label)
        
        settings_frame.columnconfigure(1, weight=1)
    
    def _create_advanced_section(self, parent):
        """Create advanced options section."""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding="10")
        advanced_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Checkbutton(advanced_frame, text="Use Grayscale Chips (with centering)",
                       variable=self.use_grayscale_chips).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Checkbutton(advanced_frame, text="Enable Pre-split Erosion",
                       variable=self.enable_pre_erosion).grid(row=1, column=0, sticky=tk.W, pady=2)
    
    def _create_canvas_section(self, parent):
        """Create drawing canvas section."""
        canvas_frame = ttk.LabelFrame(parent, text="Drawing Canvas", padding="10")
        canvas_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.canvas = tk.Canvas(canvas_frame, width=600, height=300, bg='white', 
                               relief=tk.SUNKEN, bd=2)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(3, weight=1)
        
        self._enable_drawing()
    
    def _create_right_panel(self, parent):
        """Create right processing panel."""
        right_frame = ttk.Frame(parent)
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        
        self._create_processing_tab()
        self._create_results_tab()
        self._create_statistics_tab()
    
    def _create_processing_tab(self):
        """Create processing pipeline tab."""
        processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(processing_frame, text="Processing Pipeline")
        
        self.image_labels = {}
        stages = [
            ("original", "1. Original Image"),
            ("grayscale", "2. Grayscale"),
            ("segmentation", "3. Segmentation Mask"),
            ("detected", "4. Detected Regions"),
            ("chips", "5. Digit Chips (28x28)")
        ]
        
        for i, (key, title) in enumerate(stages):
            row = i // 3
            col = i % 3
            
            stage_frame = ttk.LabelFrame(processing_frame, text=title, padding="5")
            stage_frame.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            label = ttk.Label(stage_frame, text="No image", anchor="center")
            label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.image_labels[key] = label
            
            stage_frame.columnconfigure(0, weight=1)
            stage_frame.rowconfigure(0, weight=1)
            processing_frame.columnconfigure(col, weight=1)
            processing_frame.rowconfigure(row, weight=1)
    
    def _create_results_tab(self):
        """Create results and analysis tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results & Analysis")
        
        results_label = ttk.Label(results_frame, text="Prediction Results", font=("Arial", 12, "bold"))
        results_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        text_frame = ttk.Frame(results_frame)
        text_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.results_text = tk.Text(text_frame, height=20, width=60, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
    
    def _create_statistics_tab(self):
        """Create statistics tab."""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="Statistics")
        
        self.stats_text = tk.Text(stats_frame, height=20, width=60, wrap=tk.WORD, state=tk.DISABLED)
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
    
    def _setup_callbacks(self):
        """Set up callback functions."""
        pass  # No callbacks needed - using direct assignment
    
    def _update_conf_label(self, value):
        """Update confidence threshold label."""
        self.conf_label.config(text=f"{float(value):.2f}")
    
    def _update_fg_label(self, value):
        """Update foreground threshold label."""
        self.fg_label.config(text=f"{float(value):.2f}")
    
    def _enable_drawing(self):
        """Enable drawing on canvas."""
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)
        self.canvas.focus_set()
    
    def _start_draw(self, event):
        """Start drawing."""
        self.last_xy = (event.x, event.y)
    
    def _draw(self, event):
        """Continue drawing."""
        if self.last_xy:
            self.canvas.create_line(self.last_xy[0], self.last_xy[1], event.x, event.y,
                                   fill='black', width=12, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw.line([self.last_xy[0], self.last_xy[1], event.x, event.y],
                          fill='black', width=12)
            self.last_xy = (event.x, event.y)
            self.draw_img = self.draw._image
    
    def _stop_draw(self, event):
        """Stop drawing."""
        self.last_xy = None
    
    # =========================================================================
    # IMAGE PROCESSING FUNCTIONS
    # =========================================================================
    
    def _open_image(self):
        """Open and load image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                image = cv2.imread(file_path)
                if image is None:
                    print("ERROR: Could not load image")
                    return
                
                self.current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                self._update_image_label(self.image_labels["original"], self.current_gray)
                self._update_image_label(self.image_labels["grayscale"], self.current_gray)
                
                for key in ["segmentation", "detected", "chips"]:
                    self.image_labels[key].config(text="No image")
                
                print(f"Image loaded successfully: {file_path}")
                
            except Exception as e:
                print(f"ERROR: Failed to load image: {str(e)}")
    
    def _start_processing(self):
        """Start image processing in separate thread."""
        if self.is_processing:
            return
        
        if self.current_gray is None and self.draw_img.getbbox() is None:
            print("WARNING: Please draw a digit or open an image first")
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_image)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_image(self):
        """Process current image."""
        try:
            self.last_segmentation_used = "Unknown"
            
            if self.current_gray is None:
                # Processing drawn image
                self.current_gray = np.array(self.draw_img)
                
                # Invert for display (white digits on black background)
                original_display = 255 - self.current_gray
                grayscale_display = 255 - self.current_gray
                
                self.after(0, lambda: self._update_image_label(self.image_labels["original"], original_display))
                self.after(0, lambda: self._update_image_label(self.image_labels["grayscale"], grayscale_display))
                
                # Invert for processing (black digits on white background)
                self.current_gray = 255 - self.current_gray
            
            # Perform segmentation
            segmentation_method = self.method_var.get()
            mask = segment_image_combined(self.current_gray, segmentation_method)
            self.current_mask = mask
            
            # Get segmentation method used (will be updated by segmentation module)
            self.last_segmentation_used = get_last_segmentation_method()
            
            self.after(0, lambda: self._update_image_label(self.image_labels["segmentation"], mask))
            
            # Extract digits
            chips, vis = extract_digits_combined(
                mask, self.current_gray,
                segmentation_method=self.method_var.get(),
                split_method=self.split_method_var.get(),
                use_grayscale_chips=self.use_grayscale_chips.get(),
                enable_pre_erosion=self.enable_pre_erosion.get()
            )
            
            self.digit_chips = chips
            
            self.after(0, lambda: self._update_image_label(self.image_labels["detected"], vis))
            
            if chips:
                chips_display = create_chips_display(chips)
                self.after(0, lambda: self._update_image_label(self.image_labels["chips"], chips_display))
            else:
                self.after(0, lambda: self.image_labels["chips"].config(text="No chips detected"))
            
            # Predict digits
            if chips:
                predictions, all_predictions, all_confidences = self.model_manager.predict_with_model(
                    chips, self.model_var.get())
                self.last_model_used = self.model_var.get().upper()
                self.after(0, lambda: self._display_results(predictions, all_predictions, all_confidences))
            else:
                self.after(0, lambda: self._display_results([], {}, {}))
            
            self.last_splitting_method_used = get_last_splitting_method()
            self.total_processed += 1
            if chips:
                self.successful_detections += 1
            
            self.after(0, self._update_statistics)
            self.after(0, self._enable_drawing)
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            print(f"Processing failed - check terminal for details")
            self.after(0, self._enable_drawing)
        
        finally:
            self.is_processing = False
    
    def _display_results(self, predictions, all_predictions, all_confidences):
        """Display prediction results."""
        self.results_text.delete(1.0, tk.END)
        
        if not predictions:
            self.results_text.insert(tk.END, "No digits detected or predicted.\n")
            return
        
        result_text = f"Detected {len(predictions)} digit(s):\n\n"
        for i, pred in enumerate(predictions):
            result_text += f"Digit {i+1}: {pred}\n"
        
        result_text += f"\n" + "="*50 + "\n"
        result_text += f"MODEL COMPARISON:\n"
        result_text += f"="*50 + "\n\n"
        
        for model_name in ['CNN', 'SVM', 'RF']:
            if model_name in all_predictions and model_name in all_confidences:
                model_preds = all_predictions[model_name]
                model_conf = all_confidences[model_name]
                result_text += f"{model_name}:\n"
                if len(model_preds) == 0:
                    result_text += f"  Predictions: [] (FAILED)\n"
                    result_text += f"  Confidence: {model_conf:.4f} (ERROR)\n"
                else:
                    result_text += f"  Predictions: {model_preds}\n"
                    result_text += f"  Confidence: {model_conf:.4f}\n"
                if model_name == self.last_model_used:
                    result_text += f"  → SELECTED (Best confidence)\n"
                result_text += "\n"
        
        result_text += f"Selected Model: {self.last_model_used}\n"
        result_text += f"Segmentation: {self.last_segmentation_used}\n"
        result_text += f"Splitting: {self.last_splitting_method_used}\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)
    
    def _update_image_label(self, label, image):
        """Update image label with new image."""
        if image is None:
            label.config(text="No image")
            return
        
        try:
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image, mode='L')
            else:
                pil_image = Image.fromarray(image)
            
            pil_image.thumbnail((150, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(pil_image)
            
            label.config(image=photo, text="")
            label.image = photo
            
        except Exception as e:
            print(f"Error updating image label: {e}")
            label.config(text=f"Error: {str(e)}")
    
    def _update_statistics(self):
        """Update statistics display."""
        stats_text = f"""System Statistics
==================

Processing Statistics:
• Total Images Processed: {self.total_processed}
• Successful Detections: {self.successful_detections}
• Success Rate: {(self.successful_detections/max(1, self.total_processed)*100):.1f}%

Current Settings:
• Model: {self.model_var.get().upper()} (Last used: {self.last_model_used})
• Segmentation: {self.method_var.get().title()} (Last used: {self.last_segmentation_used})
• Split Method: {self.split_method_var.get().title()} (Last used: {self.last_splitting_method_used})
• Confidence Threshold: {self.confidence_threshold.get():.2f}
• Foreground Threshold: {self.foreground_threshold.get():.2f}
• Use Grayscale Chips: {self.use_grayscale_chips.get()}
• Pre-split Erosion: {self.enable_pre_erosion.get()}

System Status:
• Processing: {'Active' if self.is_processing else 'Idle'}
• Models Loaded: {', '.join(self.model_manager.get_loaded_models())}
• Canvas: {'Active' if self.current_gray is not None else 'Empty'}

Instructions:
1. Draw digits on the canvas or open an image
2. Select your preferred settings
3. Click 'Process' to analyze the image
4. View results in the 'Results & Analysis' tab
5. Check statistics in the 'Statistics' tab
"""
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    # =========================================================================
    # UTILITY FUNCTIONS
    # =========================================================================
    
    def _clear_canvas(self):
        """Clear drawing canvas."""
        self.canvas.delete("all")
        self.draw_img = Image.new("L", (600, 500), color=255)
        self.draw = ImageDraw.Draw(self.draw_img)
        self.current_gray = None
        self.current_mask = None
        self.digit_chips = []
        
        for label in self.image_labels.values():
            label.config(text="No image", image="")
        
        self.results_text.delete(1.0, tk.END)
        self._enable_drawing()
    
    def _load_models(self):
        """Load machine learning models."""
        pass  # Models are loaded in ModelManager.__init__()
    
    def _start_auto_update(self):
        """Start automatic statistics updates."""
        self._update_statistics()
        self.after(5000, self._start_auto_update)

if __name__ == "__main__":
    app = DigitRecognitionGUI()
    app.mainloop()
