#!/usr/bin/env python3
"""
Flatmap Visualiser - Interactive image viewer with gallery and key mapping
Modern UI with drag-and-drop, 1D gallery view, and intuitive key assignment.
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QSize, QTimer, QRect, QPoint, pyqtSignal
from PyQt5.QtGui import QPixmap, QKeyEvent, QColor, QFont, QPainter, QBrush, QPen
from PyQt5.QtWidgets import QMenu


class ImageGalleryWidget(QWidget):
    """1D horizontal gallery view for image selection."""
    
    image_selected = pyqtSignal(Path)
    image_removed = pyqtSignal(Path)
    add_images = pyqtSignal()
    image_double_clicked = pyqtSignal(Path)
    default_image_changed = pyqtSignal(object)  # Emits Path or None
    
    def __init__(self):
        super().__init__()
        self.images = []  # List of Path objects
        self.key_mappings = {}  # Maps Path to key string
        self.default_image = None
        self.selected_image = None
        self.hovered_image = None
        
        self.setFixedHeight(100)
        self.setStyleSheet("background-color: #FFFFFF; border-top: 1px solid #444;")
        
        self.image_width = 80
        self.padding = 10
        self.thumb_cache = {}  # Cache for thumbnails
        
        self.setMouseTracking(True)
        
    def set_images(self, images):
        """Set the list of images."""
        self.images = images
        self.thumb_cache.clear()
        self.update()
        
    def add_image(self, image_path):
        """Add a single image."""
        if image_path not in self.images:
            self.images.append(image_path)
            self.thumb_cache.clear()
            self.update()
            
    def remove_image(self, image_path):
        """Remove an image from the gallery."""
        if image_path in self.images:
            self.images.remove(image_path)
            if self.selected_image == image_path:
                self.selected_image = None
            if self.default_image == image_path:
                self.default_image = None
            self.thumb_cache.clear()
            self.image_removed.emit(image_path)
            self.update()
            
    def set_key_mapping(self, image_path, key_str):
        """Set key mapping for an image."""
        self.key_mappings[image_path] = key_str
        self.update()
        
    def set_default_image(self, image_path):
        """Set the default image."""
        self.default_image = image_path
        self.default_image_changed.emit(image_path)
        self.update()
        
    def clear_default_image(self):
        """Clear the default image."""
        self.default_image = None
        self.default_image_changed.emit(None)
        self.update()
        
    def get_image_at_pos(self, x):
        """Get image path at x position, or None if in add button."""
        if not self.images:
            return None
            
        current_x = self.padding
        
        # Check default image first if it exists
        if self.default_image and self.default_image in self.images:
            if current_x <= x < current_x + self.image_width:
                return self.default_image
            current_x += self.image_width + self.padding
        
        # Check other images
        for img in self.images:
            if img == self.default_image:
                continue
            if current_x <= x < current_x + self.image_width:
                return img
            current_x += self.image_width + self.padding
        
        return None
        
    def paintEvent(self, event):
        """Paint the gallery."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        if not self.images:
            # Show empty state with add button
            painter.fillRect(self.rect(), QColor("#1e1e1e"))
            painter.setPen(QColor("#888"))
            painter.setFont(QFont("Arial", 12))
            painter.drawText(self.rect(), Qt.AlignCenter, 
                           "No images loaded. Click + to add images or drag & drop.")
            
            # Draw add button
            add_size = 60
            add_x = self.width() - add_size - self.padding
            add_y = (self.height() - add_size) // 2
            add_rect = QRect(add_x, add_y, add_size, add_size)
            painter.fillRect(add_rect, QColor("#333"))
            painter.setPen(QPen(QColor("#666"), 2))
            painter.drawRect(add_rect)
            painter.setPen(QColor("#888"))
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(add_rect, Qt.AlignCenter, "+")
            return
        
        current_x = self.padding
        
        # Draw default image first if exists
        if self.default_image and self.default_image in self.images:
            self.draw_image_thumbnail(painter, self.default_image, current_x, is_default=True)
            current_x += self.image_width + self.padding
        
        # Draw other images
        for img in self.images:
            if img == self.default_image:
                continue
            self.draw_image_thumbnail(painter, img, current_x)
            current_x += self.image_width + self.padding
        
        # Draw add button
        add_size = 60
        add_x = current_x + self.padding
        add_y = (self.height() - add_size) // 2
        add_rect = QRect(add_x, add_y, add_size, add_size)
        
        is_add_hovered = (self.hovered_image is None and 
                         add_x <= self.mapFromGlobal(
                             self.mapToGlobal(QPoint(0, 0))).x() + self.width())
        
        painter.fillRect(add_rect, QColor("#333" if is_add_hovered else "#2a2a2a"))
        painter.setPen(QPen(QColor("#999" if is_add_hovered else "#666"), 2))
        painter.drawRect(add_rect)
        painter.setPen(QColor("#999" if is_add_hovered else "#888"))
        painter.setFont(QFont("Arial", 24, QFont.Bold))
        painter.drawText(add_rect, Qt.AlignCenter, "+")
        
    def draw_image_thumbnail(self, painter, image_path, x, is_default=False):
        """Draw a thumbnail for an image."""
        if image_path not in self.thumb_cache:
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                thumb = pixmap.scaledToHeight(
                    self.height() - 4,
                    Qt.SmoothTransformation
                )
                self.thumb_cache[image_path] = thumb
            else:
                return
        else:
            thumb = self.thumb_cache[image_path]
        
        y = 2
        rect = QRect(x, y, self.image_width, self.height() - 4)
        
        # Draw highlight border if default or selected
        if is_default:
            painter.setPen(QPen(QColor("#FFD700"), 3))  # Gold for default
        elif self.selected_image == image_path:
            painter.setPen(QPen(QColor("#00BFFF"), 3))  # Light blue for selected
        else:
            painter.setPen(QPen(QColor("#555"), 1))
        
        painter.drawRect(rect)
        
        # Draw thumbnail
        scaled_thumb = thumb.scaledToWidth(
            self.image_width - 4,
            Qt.SmoothTransformation
        )
        thumb_y = y + (self.height() - 4 - scaled_thumb.height()) // 2
        painter.drawPixmap(x + 2, thumb_y, scaled_thumb)
        
        # Draw key mapping if exists
        if image_path in self.key_mappings:
            key_str = self.key_mappings[image_path]
            painter.setPen(QColor("#FFF"))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            key_rect = QRect(x, y, self.image_width, 20)
            painter.fillRect(key_rect, QColor(0, 0, 0, 200))
            painter.drawText(key_rect, Qt.AlignCenter, key_str)
        
        # Draw remove button (X) on hover
        if self.hovered_image == image_path:
            x_size = 20
            x_rect = QRect(x + self.image_width - x_size, y, x_size, x_size)
            painter.fillRect(x_rect, QColor(200, 50, 50, 200))
            painter.setPen(QColor("#FFF"))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            painter.drawText(x_rect, Qt.AlignCenter, "×")
    
    def mouseMoveEvent(self, event):
        """Track mouse movement for hover effects."""
        self.hovered_image = self.get_image_at_pos(event.x())
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks."""
        image = self.get_image_at_pos(event.x())
        
        if event.button() == Qt.LeftButton:
            if image is None:
                # Check if add button was clicked
                add_size = 60
                add_x = self.padding + len(self.images) * (self.image_width + self.padding) + self.padding
                if event.x() > add_x - add_size:
                    self.add_images.emit()
            elif event.x() > self.get_image_rect(image).right() - 20:
                # Remove button clicked
                self.remove_image(image)
            else:
                self.selected_image = image
                self.image_selected.emit(image)
                self.update()
        
        elif event.button() == Qt.RightButton:
            if image:
                menu = QMenu(self)
                
                if image != self.default_image:
                    menu.addAction("Use as Default", 
                                 lambda: self.set_default_image(image))
                else:
                    menu.addAction("Remove as Default", self.clear_default_image)
                
                menu.addAction("Assign Key", lambda: self.image_double_clicked.emit(image))
                
                menu.exec_(self.mapToGlobal(event.pos()))
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click for key assignment."""
        image = self.get_image_at_pos(event.x())
        if image and event.x() <= self.get_image_rect(image).right() - 20:
            self.image_double_clicked.emit(image)
    
    def get_image_rect(self, image_path):
        """Get the QRect for an image in the gallery."""
        if image_path not in self.images:
            return QRect()
        
        current_x = self.padding
        
        if self.default_image and self.default_image in self.images:
            if image_path == self.default_image:
                return QRect(current_x, 2, self.image_width, self.height() - 4)
            current_x += self.image_width + self.padding
        
        for img in self.images:
            if img == self.default_image:
                continue
            if img == image_path:
                return QRect(current_x, 2, self.image_width, self.height() - 4)
            current_x += self.image_width + self.padding
        
        return QRect()
    
    def leaveEvent(self, event):
        """Clear hover when leaving widget."""
        self.hovered_image = None
        self.update()


class FlatmapVisualiser(QMainWindow):
    def __init__(self, outputs_dir):
        super().__init__()
        self.outputs_dir = Path(outputs_dir)
        self.images = []
        self.current_image = None
        self.key_mappings = {}  # Maps Qt.Key_* to image path
        self.default_image = None
        self.pressed_keys = set()
        self.gallery_visible = False
        self.gallery_timer = QTimer()
        self.gallery_timer.timeout.connect(self.hide_gallery)
        self.assigning_key_for_image = None  # Track key assignment mode
        self.original_pixmap = None  # Store original pixmap during key assignment
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Flatmap Visualiser")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #FFFFFF; }")
        self.image_label.setScaledContents(False)
        main_layout.addWidget(self.image_label)
        
        # Gallery at bottom
        self.gallery = ImageGalleryWidget()
        self.gallery.image_selected.connect(self.on_image_selected)
        self.gallery.image_removed.connect(self.on_image_removed)
        self.gallery.add_images.connect(self.add_images_dialog)
        self.gallery.image_double_clicked.connect(self.assign_key_to_image)
        self.gallery.default_image_changed.connect(self.on_default_image_changed)
        main_layout.addWidget(self.gallery)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
        # Show startup screen
        self.show_startup_screen()
    
    def show_startup_screen(self):
        """Show startup screen with drag and drop support."""
        self.image_label.setText(
            "Drag and drop images here\nor click to select images"
        )
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { font-size: 18px; color: #888; background-color: #FFFFFF; }"
        )
    
    def add_images_dialog(self):
        """Open file dialog to add images."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            str(self.outputs_dir),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)"
        )
        
        for file in files:
            self.add_image(Path(file))
    
    def add_image(self, image_path: Path):
        """Add an image to the gallery."""
        if image_path not in self.images:
            self.images.append(image_path)
            self.gallery.add_image(image_path)
            
            # If this is the first image, display it
            if len(self.images) == 1:
                self.display_image(image_path)
    
    def on_image_selected(self, image_path: Path):
        """Handle image selection from gallery."""
        self.display_image(image_path)
        self.show_gallery()
    
    def on_image_removed(self, image_path: Path):
        """Handle image removal from gallery."""
        if image_path in self.images:
            self.images.remove(image_path)
        
        # If last image was removed, show startup screen
        if not self.images:
            self.show_startup_screen()
            self.gallery.set_images([])
    
    def on_default_image_changed(self, image_path):
        """Handle default image change from gallery."""
        self.default_image = image_path
        if image_path:
            self.display_image(image_path)
    
    def display_image(self, img_path: Path):
        """Display an image in the image label."""
        if not img_path.exists():
            QMessageBox.warning(self, "Error", f"Image not found: {img_path}")
            return
        
        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load: {img_path.name}")
            return
        
        # Scale image to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.current_image = img_path
        self.gallery.selected_image = img_path
        self.gallery.update()
    
    def assign_key_to_image(self, image_path: Path):
        """Enter key assignment mode for an image."""
        self.assigning_key_for_image = image_path
        
        # Display the image if not already shown
        if self.current_image != image_path:
            self.display_image(image_path)
        
        # Add overlay message to show we're waiting for a key
        self.show_key_assignment_overlay()
    
    def show_gallery(self):
        """Show the gallery and reset hide timer."""
        self.gallery_visible = True
        self.gallery.show()
        
        # Reset timer to hide gallery after inactivity
        self.gallery_timer.stop()
        self.gallery_timer.start(3000)  # Hide after 3 seconds of inactivity
    
    def hide_gallery(self):
        """Hide the gallery."""
        self.gallery_visible = False
        self.gallery_timer.stop()
    
    def show_key_assignment_overlay(self):
        """Show overlay message for key assignment."""
        if not self.current_image or not self.current_image.exists():
            return
        
        # Store the original pixmap
        pixmap = QPixmap(str(self.current_image))
        if pixmap.isNull():
            return
        
        # Create a copy to draw on
        overlay_pixmap = pixmap.copy()
        painter = QPainter(overlay_pixmap)
        
        # Semi-transparent overlay
        painter.fillRect(overlay_pixmap.rect(), QColor(0, 0, 0, 180))
        
        # Draw text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 32, QFont.Bold)
        painter.setFont(font)
        text = "Press any key to assign..."
        painter.drawText(overlay_pixmap.rect(), Qt.AlignCenter, text)
        
        # Draw image name
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Arial", 16)
        painter.setFont(font)
        image_name = f"Image: {self.assigning_key_for_image.name}"
        text_rect = overlay_pixmap.rect()
        text_rect.adjust(0, 60, 0, 0)
        painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, image_name)
        
        painter.end()
        
        # Display the overlay
        scaled_pixmap = overlay_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def handle_key_assignment(self, key):
        """Handle key assignment for the current image."""
        if not self.assigning_key_for_image:
            return
        
        image_path = self.assigning_key_for_image
        key_str = self.key_to_string(key)
        
        # Check if key is already mapped
        if key in self.key_mappings:
            existing_image = self.key_mappings[key]
            
            # Ask for confirmation to replace
            reply = QMessageBox.question(
                self,
                "Key Already Mapped",
                f"Key '{key_str}' is already mapped to:\n{existing_image.name}\n\n"
                f"Replace with:\n{image_path.name}?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel
            )
            
            if reply != QMessageBox.Yes:
                # Cancel - restore original display
                self.assigning_key_for_image = None
                self.display_image(image_path)
                return
        
        # Assign the key
        self.key_mappings[key] = image_path
        self.gallery.set_key_mapping(image_path, key_str)
        
        # Exit assignment mode
        self.assigning_key_for_image = None
        
        # Restore original display
        self.display_image(image_path)
        self.show_gallery()
    
    def mouseMoveEvent(self, event):
        """Show gallery on mouse movement."""
        if self.images and not self.gallery_visible:
            self.show_gallery()
        elif self.gallery_visible:
            # Reset timer on mouse movement
            self.gallery_timer.stop()
            self.gallery_timer.start(3000)
    
    def dragEnterEvent(self, event):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop event for drag and drop."""
        for url in event.mimeData().urls():
            file_path = Path(url.toLocalFile())
            if file_path.is_file() and file_path.suffix.lower() in {
                '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'
            }:
                self.add_image(file_path)
        
        self.show_gallery()
        event.acceptProposedAction()
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        key = event.key()
        
        # Ignore modifier keys and repeated events
        if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
            return
        
        if event.isAutoRepeat():
            return
        
        # If in key assignment mode, assign the key
        if self.assigning_key_for_image:
            self.handle_key_assignment(key)
            return
        
        # Check if this key is mapped
        if key in self.key_mappings:
            self.pressed_keys.add(key)
            img_path = self.key_mappings[key]
            self.display_image(img_path)
            self.show_gallery()
    
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
    visualiser.showMaximized()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
