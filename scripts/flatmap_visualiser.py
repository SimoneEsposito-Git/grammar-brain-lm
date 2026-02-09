#!/usr/bin/env python3
"""
Flatmap Visualiser - Interactive image viewer with keyboard shortcuts
Allows mapping images to keyboard keys with optional default image on key release.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QListWidget, QFileDialog,
    QGroupBox, QGridLayout, QScrollArea, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QKeyEvent


class FlatmapVisualiser(QMainWindow):
    def __init__(self, outputs_dir):
        super().__init__()
        self.outputs_dir = Path(outputs_dir)
        self.key_mappings = {}  # Maps Qt.Key_* to image path
        self.default_image = None
        self.current_image = None
        self.pressed_keys = set()  # Track currently pressed keys
        
        self.init_ui()
        self.scan_images()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Flatmap Visualiser")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel: Image display
        left_panel = self.create_image_panel()
        main_layout.addWidget(left_panel, 3)
        
        # Right panel: Controls
        right_panel = self.create_control_panel()
        main_layout.addWidget(right_panel, 1)
        
    def create_image_panel(self):
        """Create the image display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px solid #555; }")
        self.image_label.setScaledContents(False)
        
        # Scroll area for large images
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Current image info
        self.info_label = QLabel("No image loaded")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("QLabel { font-size: 12px; padding: 5px; }")
        layout.addWidget(self.info_label)
        
        return panel
        
    def create_control_panel(self):
        """Create the control panel with mapping options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Directory selector
        dir_group = QGroupBox("Image Directory")
        dir_layout = QVBoxLayout()
        self.dir_label = QLabel(str(self.outputs_dir))
        self.dir_label.setWordWrap(True)
        self.dir_label.setStyleSheet("QLabel { padding: 5px; background-color: #f0f0f0; }")
        dir_layout.addWidget(self.dir_label)
        change_dir_btn = QPushButton("Change Directory")
        change_dir_btn.clicked.connect(self.change_directory)
        dir_layout.addWidget(change_dir_btn)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # Available images
        images_group = QGroupBox("Available Images")
        images_layout = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.preview_image)
        images_layout.addWidget(self.image_list)
        refresh_btn = QPushButton("Refresh Images")
        refresh_btn.clicked.connect(self.scan_images)
        images_layout.addWidget(refresh_btn)
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)
        
        # Default image selector
        default_group = QGroupBox("Default Image")
        default_layout = QVBoxLayout()
        self.default_combo = QComboBox()
        self.default_combo.addItem("(None)")
        default_layout.addWidget(self.default_combo)
        set_default_btn = QPushButton("Set as Default")
        set_default_btn.clicked.connect(self.set_default_image)
        default_layout.addWidget(set_default_btn)
        clear_default_btn = QPushButton("Clear Default")
        clear_default_btn.clicked.connect(self.clear_default_image)
        default_layout.addWidget(clear_default_btn)
        default_group.setLayout(default_layout)
        layout.addWidget(default_group)
        
        # Key mapping
        mapping_group = QGroupBox("Key Mappings")
        mapping_layout = QVBoxLayout()
        
        map_instruction = QLabel("Select image, press key to map:")
        map_instruction.setWordWrap(True)
        mapping_layout.addWidget(map_instruction)
        
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("Press a key...")
        self.key_input.setReadOnly(True)
        mapping_layout.addWidget(self.key_input)
        
        map_btn = QPushButton("Map Selected Image to Key")
        map_btn.clicked.connect(self.map_image_to_key)
        mapping_layout.addWidget(map_btn)
        
        # Current mappings display
        self.mappings_list = QListWidget()
        mapping_layout.addWidget(QLabel("Current Mappings:"))
        mapping_layout.addWidget(self.mappings_list)
        
        clear_mapping_btn = QPushButton("Clear Selected Mapping")
        clear_mapping_btn.clicked.connect(self.clear_mapping)
        mapping_layout.addWidget(clear_mapping_btn)
        
        mapping_group.setLayout(mapping_layout)
        layout.addWidget(mapping_group)
        
        layout.addStretch()
        
        return panel
        
    def scan_images(self):
        """Scan the outputs directory for image files."""
        self.image_list.clear()
        self.default_combo.clear()
        self.default_combo.addItem("(None)")
        
        if not self.outputs_dir.exists():
            QMessageBox.warning(self, "Directory Error", 
                              f"Directory does not exist:\n{self.outputs_dir}")
            return
            
        # Supported image formats
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        
        # Recursively find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.outputs_dir.rglob(f'*{ext}'))
            image_files.extend(self.outputs_dir.rglob(f'*{ext.upper()}'))
        
        # Sort and add to list
        image_files = sorted(set(image_files))
        for img_path in image_files:
            rel_path = img_path.relative_to(self.outputs_dir)
            self.image_list.addItem(str(rel_path))
            self.default_combo.addItem(str(rel_path))
            
        self.info_label.setText(f"Found {len(image_files)} images")
        
    def change_directory(self):
        """Change the outputs directory."""
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select Images Directory", str(self.outputs_dir)
        )
        if new_dir:
            self.outputs_dir = Path(new_dir)
            self.dir_label.setText(str(self.outputs_dir))
            self.scan_images()
            self.key_mappings.clear()
            self.update_mappings_display()
            
    def preview_image(self, item):
        """Preview the selected image."""
        rel_path = item.text()
        img_path = self.outputs_dir / rel_path
        self.display_image(img_path)
        
    def display_image(self, img_path):
        """Display an image in the image label."""
        if not img_path.exists():
            self.info_label.setText(f"Image not found: {img_path.name}")
            return
            
        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            self.info_label.setText(f"Failed to load: {img_path.name}")
            return
            
        # Scale image to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.current_image = img_path
        
        rel_path = img_path.relative_to(self.outputs_dir)
        self.info_label.setText(f"Viewing: {rel_path}")
        
    def set_default_image(self):
        """Set the default image from combo box."""
        selected = self.default_combo.currentText()
        if selected == "(None)":
            self.default_image = None
            QMessageBox.information(self, "Default Image", "No default image set")
        else:
            self.default_image = self.outputs_dir / selected
            QMessageBox.information(self, "Default Image", 
                                  f"Default set to:\n{selected}")
            
    def clear_default_image(self):
        """Clear the default image."""
        self.default_image = None
        self.default_combo.setCurrentIndex(0)
        QMessageBox.information(self, "Default Image", "Default image cleared")
        
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        key = event.key()
        
        # Ignore modifier keys and repeated events
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            return
            
        if event.isAutoRepeat():
            return
            
        # Check if this key is mapped
        if key in self.key_mappings:
            self.pressed_keys.add(key)
            img_path = self.key_mappings[key]
            self.display_image(img_path)
        else:
            # Store the key for mapping
            self.key_input.setText(self.key_to_string(key))
            
    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release events."""
        key = event.key()
        
        if event.isAutoRepeat():
            return
            
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)
            
            # If no keys are pressed and we have a default, show it
            if not self.pressed_keys and self.default_image:
                self.display_image(self.default_image)
                
    def map_image_to_key(self):
        """Map the selected image to the entered key."""
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an image first")
            return
            
        key_text = self.key_input.text()
        if not key_text:
            QMessageBox.warning(self, "No Key", "Please press a key first")
            return
            
        # Convert key text back to key code
        key = self.string_to_key(key_text)
        if key is None:
            QMessageBox.warning(self, "Invalid Key", "Could not map this key")
            return
            
        rel_path = selected_items[0].text()
        img_path = self.outputs_dir / rel_path
        
        self.key_mappings[key] = img_path
        self.update_mappings_display()
        self.key_input.clear()
        
        QMessageBox.information(self, "Mapping Created", 
                              f"'{key_text}' -> {rel_path}")
        
    def clear_mapping(self):
        """Clear the selected key mapping."""
        selected_items = self.mappings_list.selectedItems()
        if not selected_items:
            return
            
        mapping_text = selected_items[0].text()
        key_text = mapping_text.split(" -> ")[0]
        
        # Find and remove the mapping
        key_to_remove = None
        for key, path in self.key_mappings.items():
            if self.key_to_string(key) == key_text:
                key_to_remove = key
                break
                
        if key_to_remove:
            del self.key_mappings[key_to_remove]
            self.update_mappings_display()
            
    def update_mappings_display(self):
        """Update the display of current mappings."""
        self.mappings_list.clear()
        for key, path in sorted(self.key_mappings.items()):
            rel_path = path.relative_to(self.outputs_dir)
            key_str = self.key_to_string(key)
            self.mappings_list.addItem(f"{key_str} -> {rel_path}")
            
    @staticmethod
    def key_to_string(key):
        """Convert Qt key code to readable string."""
        # Handle alphanumeric keys
        if Qt.Key_A <= key <= Qt.Key_Z:
            return chr(key)
        elif Qt.Key_0 <= key <= Qt.Key_9:
            return chr(key)
        else:
            # Handle special keys
            key_names = {
                Qt.Key_Space: "Space",
                Qt.Key_Enter: "Enter",
                Qt.Key_Return: "Return",
                Qt.Key_Tab: "Tab",
                Qt.Key_Escape: "Esc",
                Qt.Key_Left: "Left",
                Qt.Key_Right: "Right",
                Qt.Key_Up: "Up",
                Qt.Key_Down: "Down",
                Qt.Key_F1: "F1", Qt.Key_F2: "F2", Qt.Key_F3: "F3",
                Qt.Key_F4: "F4", Qt.Key_F5: "F5", Qt.Key_F6: "F6",
                Qt.Key_F7: "F7", Qt.Key_F8: "F8", Qt.Key_F9: "F9",
                Qt.Key_F10: "F10", Qt.Key_F11: "F11", Qt.Key_F12: "F12",
            }
            return key_names.get(key, f"Key_{key}")
            
    @staticmethod
    def string_to_key(key_str):
        """Convert string to Qt key code."""
        if len(key_str) == 1 and key_str.isalpha():
            return ord(key_str.upper())
        elif len(key_str) == 1 and key_str.isdigit():
            return ord(key_str)
        else:
            key_map = {
                "Space": Qt.Key_Space,
                "Enter": Qt.Key_Enter,
                "Return": Qt.Key_Return,
                "Tab": Qt.Key_Tab,
                "Esc": Qt.Key_Escape,
                "Left": Qt.Key_Left,
                "Right": Qt.Key_Right,
                "Up": Qt.Key_Up,
                "Down": Qt.Key_Down,
                "F1": Qt.Key_F1, "F2": Qt.Key_F2, "F3": Qt.Key_F3,
                "F4": Qt.Key_F4, "F5": Qt.Key_F5, "F6": Qt.Key_F6,
                "F7": Qt.Key_F7, "F8": Qt.Key_F8, "F9": Qt.Key_F9,
                "F10": Qt.Key_F10, "F11": Qt.Key_F11, "F12": Qt.Key_F12,
            }
            return key_map.get(key_str)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flatmap Visualiser")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="../outputs",
        help="Path to outputs directory (default: ../outputs)"
    )
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Resolve the directory path
    outputs_dir = Path(__file__).parent / args.dir
    outputs_dir = outputs_dir.resolve()
    
    visualiser = FlatmapVisualiser(outputs_dir)
    visualiser.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
