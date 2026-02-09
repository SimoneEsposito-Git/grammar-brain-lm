#!/usr/bin/env python3
"""
Flatmap Visualiser - Interactive image viewer with keyboard shortcuts (Tkinter version)
Allows mapping images to keyboard keys with optional default image on key release.
"""

import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


class FlatmapVisualiser:
    def __init__(self, root, outputs_dir):
        self.root = root
        self.outputs_dir = Path(outputs_dir)
        self.key_mappings = {}  # Maps key string to image path
        self.default_image = None
        self.current_image = None
        self.pressed_keys = set()  # Track currently pressed keys
        self.photo_image = None  # Keep reference to prevent garbage collection
        self.pending_key = None  # Store key waiting to be mapped
        
        self.init_ui()
        self.scan_images()
        
        # Bind keyboard events globally
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.root.title("Flatmap Visualiser")
        self.root.geometry("1400x900")
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Image display
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Image canvas with scrollbars
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(canvas_frame, bg='#2b2b2b', highlightthickness=2, 
                                      highlightbackground='#555')
        self.h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, 
                                        command=self.image_canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, 
                                        command=self.image_canvas.yview)
        
        self.image_canvas.configure(xscrollcommand=self.h_scrollbar.set,
                                   yscrollcommand=self.v_scrollbar.set)
        
        self.image_canvas.grid(row=0, column=0, sticky='nsew')
        self.h_scrollbar.grid(row=1, column=0, sticky='ew')
        self.v_scrollbar.grid(row=0, column=1, sticky='ns')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Info label
        self.info_label = ttk.Label(left_frame, text="No image loaded", 
                                   anchor=tk.CENTER, relief=tk.SUNKEN)
        self.info_label.pack(fill=tk.X, pady=(5, 0))
        
        # Right panel: Controls
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        self.create_control_panel(right_frame)
        
    def create_control_panel(self, parent):
        """Create the control panel with mapping options."""
        # Canvas for scrolling
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Directory selector
        dir_frame = ttk.LabelFrame(scrollable_frame, text="Image Directory", padding=10)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.dir_label = ttk.Label(dir_frame, text=str(self.outputs_dir), 
                                   wraplength=300, background='#f0f0f0', 
                                   relief=tk.SUNKEN, padding=5)
        self.dir_label.pack(fill=tk.X)
        
        ttk.Button(dir_frame, text="Change Directory", 
                  command=self.change_directory).pack(fill=tk.X, pady=(5, 0))
        
        # Available images
        images_frame = ttk.LabelFrame(scrollable_frame, text="Available Images", padding=10)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        list_frame = ttk.Frame(images_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        list_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.image_listbox = tk.Listbox(list_frame, yscrollcommand=list_scrollbar.set, 
                                        height=10, selectmode=tk.SINGLE)
        list_scrollbar.config(command=self.image_listbox.yview)
        
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.image_listbox.bind('<Double-Button-1>', self.preview_image)
        
        ttk.Button(images_frame, text="Refresh Images", 
                  command=self.scan_images).pack(fill=tk.X, pady=(5, 0))
        
        # Default image selector
        default_frame = ttk.LabelFrame(scrollable_frame, text="Default Image", padding=10)
        default_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.default_var = tk.StringVar(value="(None)")
        self.default_combo = ttk.Combobox(default_frame, textvariable=self.default_var, 
                                         state='readonly')
        self.default_combo['values'] = ['(None)']
        self.default_combo.pack(fill=tk.X)
        
        btn_frame = ttk.Frame(default_frame)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(btn_frame, text="Set as Default", 
                  command=self.set_default_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_frame, text="Clear Default", 
                  command=self.clear_default_image).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Key mapping
        mapping_frame = ttk.LabelFrame(scrollable_frame, text="Key Mappings", padding=10)
        mapping_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        ttk.Label(mapping_frame, text="Select image, press key to map:", 
                 wraplength=280).pack()
        
        self.key_entry = ttk.Entry(mapping_frame, state='readonly')
        self.key_entry.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(mapping_frame, text="Map Selected Image to Key", 
                  command=self.map_image_to_key).pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(mapping_frame, text="Current Mappings:").pack(pady=(10, 5))
        
        # Mappings listbox
        map_list_frame = ttk.Frame(mapping_frame)
        map_list_frame.pack(fill=tk.BOTH, expand=True)
        
        map_scrollbar = ttk.Scrollbar(map_list_frame, orient=tk.VERTICAL)
        self.mappings_listbox = tk.Listbox(map_list_frame, yscrollcommand=map_scrollbar.set, 
                                           height=8, selectmode=tk.SINGLE)
        map_scrollbar.config(command=self.mappings_listbox.yview)
        
        self.mappings_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        map_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(mapping_frame, text="Clear Selected Mapping", 
                  command=self.clear_mapping).pack(fill=tk.X, pady=(5, 0))
        
        # Pack scrollbar and canvas
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def scan_images(self):
        """Scan the outputs directory for image files."""
        self.image_listbox.delete(0, tk.END)
        self.default_combo['values'] = ['(None)']
        
        if not self.outputs_dir.exists():
            messagebox.showwarning("Directory Error", 
                                  f"Directory does not exist:\n{self.outputs_dir}")
            return
            
        # Supported image formats
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
        
        # Recursively find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.outputs_dir.rglob(f'*{ext}'))
            image_files.extend(self.outputs_dir.rglob(f'*{ext.upper()}'))
        
        # Sort and add to list
        image_files = sorted(set(image_files))
        image_names = []
        for img_path in image_files:
            rel_path = str(img_path.relative_to(self.outputs_dir))
            self.image_listbox.insert(tk.END, rel_path)
            image_names.append(rel_path)
        
        self.default_combo['values'] = ['(None)'] + image_names
        self.info_label.config(text=f"Found {len(image_files)} images")
        
    def change_directory(self):
        """Change the outputs directory."""
        new_dir = filedialog.askdirectory(
            title="Select Images Directory", 
            initialdir=str(self.outputs_dir)
        )
        if new_dir:
            self.outputs_dir = Path(new_dir)
            self.dir_label.config(text=str(self.outputs_dir))
            self.scan_images()
            self.key_mappings.clear()
            self.update_mappings_display()
            
    def preview_image(self, event=None):
        """Preview the selected image."""
        selection = self.image_listbox.curselection()
        if not selection:
            return
            
        rel_path = self.image_listbox.get(selection[0])
        img_path = self.outputs_dir / rel_path
        self.display_image(img_path)
        
    def display_image(self, img_path):
        """Display an image on the canvas."""
        if not img_path.exists():
            self.info_label.config(text=f"Image not found: {img_path.name}")
            return
            
        try:
            # Load image with PIL
            image = Image.open(str(img_path))
            
            # Get canvas size
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            
            # Use reasonable defaults if canvas not drawn yet
            if canvas_width <= 1:
                canvas_width = 800
            if canvas_height <= 1:
                canvas_height = 600
            
            # Calculate scaling to fit canvas while maintaining aspect ratio
            img_width, img_height = image.size
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h, 1.0)  # Don't upscale
            
            if scale < 1.0:
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(image)
            
            # Clear canvas and display image
            self.image_canvas.delete("all")
            self.image_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                image=self.photo_image, anchor=tk.CENTER
            )
            
            # Update scroll region
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
            
            self.current_image = img_path
            rel_path = img_path.relative_to(self.outputs_dir)
            self.info_label.config(text=f"Viewing: {rel_path}")
            
        except Exception as e:
            self.info_label.config(text=f"Failed to load: {img_path.name}")
            messagebox.showerror("Image Error", f"Could not load image:\n{str(e)}")
            
    def set_default_image(self):
        """Set the default image from combo box."""
        selected = self.default_var.get()
        if selected == "(None)":
            self.default_image = None
            messagebox.showinfo("Default Image", "No default image set")
        else:
            self.default_image = self.outputs_dir / selected
            messagebox.showinfo("Default Image", f"Default set to:\n{selected}")
            
    def clear_default_image(self):
        """Clear the default image."""
        self.default_image = None
        self.default_var.set("(None)")
        messagebox.showinfo("Default Image", "Default image cleared")
        
    def on_key_press(self, event):
        """Handle key press events."""
        key_str = self.normalize_key(event.keysym)
        
        # Ignore modifier keys
        if key_str in ('Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                      'Alt_L', 'Alt_R', 'Super_L', 'Super_R'):
            return
            
        # If we're in mapping mode (focused on key entry), store the key
        if self.root.focus_get() == self.key_entry or self.pending_key is not None:
            self.pending_key = key_str
            self.key_entry.config(state='normal')
            self.key_entry.delete(0, tk.END)
            self.key_entry.insert(0, key_str)
            self.key_entry.config(state='readonly')
            return
            
        # Check if this key is mapped
        if key_str in self.key_mappings:
            if key_str not in self.pressed_keys:
                self.pressed_keys.add(key_str)
                img_path = self.key_mappings[key_str]
                self.display_image(img_path)
                
    def on_key_release(self, event):
        """Handle key release events."""
        key_str = self.normalize_key(event.keysym)
        
        if key_str in self.pressed_keys:
            self.pressed_keys.discard(key_str)
            
            # If no keys are pressed and we have a default, show it
            if not self.pressed_keys and self.default_image:
                self.display_image(self.default_image)
                
    def normalize_key(self, keysym):
        """Normalize key symbols to consistent format."""
        # Convert common keys to simpler names
        key_map = {
            'Return': 'Enter',
            'KP_Enter': 'Enter',
        }
        return key_map.get(keysym, keysym)
        
    def map_image_to_key(self):
        """Map the selected image to the entered key."""
        selection = self.image_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an image first")
            return
            
        key_str = self.pending_key or self.key_entry.get()
        if not key_str:
            messagebox.showwarning("No Key", "Please press a key first")
            return
            
        rel_path = self.image_listbox.get(selection[0])
        img_path = self.outputs_dir / rel_path
        
        self.key_mappings[key_str] = img_path
        self.update_mappings_display()
        
        # Clear the key entry
        self.key_entry.config(state='normal')
        self.key_entry.delete(0, tk.END)
        self.key_entry.config(state='readonly')
        self.pending_key = None
        
        messagebox.showinfo("Mapping Created", f"'{key_str}' -> {rel_path}")
        
    def clear_mapping(self):
        """Clear the selected key mapping."""
        selection = self.mappings_listbox.curselection()
        if not selection:
            return
            
        mapping_text = self.mappings_listbox.get(selection[0])
        key_str = mapping_text.split(" -> ")[0]
        
        if key_str in self.key_mappings:
            del self.key_mappings[key_str]
            self.update_mappings_display()
            
    def update_mappings_display(self):
        """Update the display of current mappings."""
        self.mappings_listbox.delete(0, tk.END)
        for key_str in sorted(self.key_mappings.keys()):
            path = self.key_mappings[key_str]
            rel_path = path.relative_to(self.outputs_dir)
            self.mappings_listbox.insert(tk.END, f"{key_str} -> {rel_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flatmap Visualiser (Tkinter)")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="../outputs",
        help="Path to outputs directory (default: ../outputs)"
    )
    args = parser.parse_args()
    
    # Resolve the directory path
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / args.dir
    outputs_dir = outputs_dir.resolve()
    
    root = tk.Tk()
    app = FlatmapVisualiser(root, outputs_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
