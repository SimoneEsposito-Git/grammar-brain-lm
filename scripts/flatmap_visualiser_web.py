#!/usr/bin/env python3
"""
Flatmap Visualiser - Web-based interactive image viewer with keyboard shortcuts
Allows mapping images to keyboard keys with optional default image on key release.
Accessible via web browser - perfect for remote/headless servers.
"""

from flask import Flask, render_template, jsonify, send_file, request
from pathlib import Path
import json
import os

app = Flask(__name__)

# Global state
config = {
    'outputs_dir': None,
    'key_mappings': {},
    'default_image': None,
    'image_files': []
}


def scan_images(outputs_dir):
    """Scan directory for image files."""
    if not outputs_dir.exists():
        return []
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(outputs_dir.rglob(f'*{ext}'))
        image_files.extend(outputs_dir.rglob(f'*{ext.upper()}'))
    
    # Convert to relative paths and sort
    rel_paths = sorted([str(f.relative_to(outputs_dir)) for f in set(image_files)])
    return rel_paths


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/images')
def get_images():
    """Get list of available images."""
    config['image_files'] = scan_images(config['outputs_dir'])
    return jsonify({
        'images': config['image_files'],
        'outputs_dir': str(config['outputs_dir'])
    })


@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Serve an image file."""
    full_path = config['outputs_dir'] / image_path
    if full_path.exists() and full_path.is_file():
        return send_file(str(full_path))
    return jsonify({'error': 'Image not found'}), 404


@app.route('/api/mappings', methods=['GET', 'POST', 'DELETE'])
def handle_mappings():
    """Handle key mappings CRUD operations."""
    if request.method == 'GET':
        return jsonify({
            'mappings': config['key_mappings'],
            'default_image': config['default_image']
        })
    
    elif request.method == 'POST':
        data = request.json
        key = data.get('key')
        image = data.get('image')
        
        if key and image:
            config['key_mappings'][key] = image
            return jsonify({'success': True, 'mappings': config['key_mappings']})
        return jsonify({'error': 'Missing key or image'}), 400
    
    elif request.method == 'DELETE':
        data = request.json
        key = data.get('key')
        
        if key and key in config['key_mappings']:
            del config['key_mappings'][key]
            return jsonify({'success': True, 'mappings': config['key_mappings']})
        return jsonify({'error': 'Key not found'}), 404


@app.route('/api/default', methods=['POST', 'DELETE'])
def handle_default():
    """Handle default image setting."""
    if request.method == 'POST':
        data = request.json
        image = data.get('image')
        config['default_image'] = image
        return jsonify({'success': True, 'default_image': config['default_image']})
    
    elif request.method == 'DELETE':
        config['default_image'] = None
        return jsonify({'success': True, 'default_image': None})


# HTML template
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flatmap Visualiser</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1e1e1e;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        .main-panel {
            flex: 3;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        
        .image-container {
            flex: 1;
            background: #2b2b2b;
            border: 2px solid #555;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: auto;
            position: relative;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        
        .info-bar {
            margin-top: 10px;
            padding: 10px;
            background: #2b2b2b;
            border-radius: 4px;
            text-align: center;
            font-size: 14px;
        }
        
        .control-panel {
            flex: 1;
            background: #252525;
            padding: 20px;
            overflow-y: auto;
            border-left: 2px solid #555;
        }
        
        .section {
            margin-bottom: 25px;
            background: #2b2b2b;
            padding: 15px;
            border-radius: 8px;
        }
        
        .section h3 {
            margin-bottom: 12px;
            color: #4CAF50;
            font-size: 16px;
            border-bottom: 1px solid #444;
            padding-bottom: 8px;
        }
        
        select, input, button {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
            background: #1e1e1e;
            border: 1px solid #555;
            color: #e0e0e0;
            border-radius: 4px;
            font-size: 14px;
        }
        
        select {
            cursor: pointer;
        }
        
        select[multiple] {
            height: 150px;
        }
        
        button {
            background: #4CAF50;
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.3s;
        }
        
        button:hover {
            background: #45a049;
        }
        
        button.secondary {
            background: #666;
        }
        
        button.secondary:hover {
            background: #777;
        }
        
        button.danger {
            background: #f44336;
        }
        
        button.danger:hover {
            background: #da190b;
        }
        
        .button-group {
            display: flex;
            gap: 5px;
        }
        
        .button-group button {
            flex: 1;
        }
        
        .mapping-item {
            padding: 8px;
            margin: 5px 0;
            background: #1e1e1e;
            border: 1px solid #555;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 13px;
        }
        
        .mapping-item:hover {
            background: #333;
        }
        
        .mapping-item.selected {
            background: #4CAF50;
            border-color: #4CAF50;
        }
        
        .key-badge {
            display: inline-block;
            background: #555;
            padding: 3px 8px;
            border-radius: 3px;
            font-family: monospace;
            font-weight: bold;
            margin-right: 8px;
        }
        
        .instructions {
            font-size: 12px;
            color: #aaa;
            margin-top: 8px;
            font-style: italic;
        }
        
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            display: none;
            animation: slideIn 0.3s;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .no-display {
            color: #666;
            text-align: center;
            padding: 40px 20px;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-panel">
            <div class="image-container" id="imageContainer">
                <div class="no-display">No image loaded - Double-click an image or press a mapped key</div>
            </div>
            <div class="info-bar" id="infoBar">Ready</div>
        </div>
        
        <div class="control-panel">
            <div class="section">
                <h3>📁 Image Directory</h3>
                <div id="directoryPath" style="font-size: 12px; word-break: break-all; margin-bottom: 10px;"></div>
                <button onclick="refreshImages()">🔄 Refresh Images</button>
            </div>
            
            <div class="section">
                <h3>🖼️ Available Images</h3>
                <select id="imageList" size="8" onchange="onImageSelect()" ondblclick="previewSelectedImage()">
                </select>
                <div class="instructions">Double-click to preview</div>
            </div>
            
            <div class="section">
                <h3>⭐ Default Image</h3>
                <select id="defaultImageSelect">
                    <option value="">(None)</option>
                </select>
                <div class="button-group">
                    <button onclick="setDefaultImage()">Set Default</button>
                    <button class="secondary" onclick="clearDefaultImage()">Clear</button>
                </div>
            </div>
            
            <div class="section">
                <h3>⌨️ Key Mappings</h3>
                <input type="text" id="keyInput" placeholder="Press a key..." readonly>
                <button onclick="mapImageToKey()">Map Selected Image</button>
                <div style="margin-top: 15px;">
                    <strong>Current Mappings:</strong>
                    <div id="mappingsList" style="max-height: 200px; overflow-y: auto; margin-top: 8px;"></div>
                </div>
                <button class="danger" onclick="clearSelectedMapping()" style="margin-top: 10px;">Clear Selected</button>
                <div class="instructions">Select image, press key, click "Map Selected Image"</div>
            </div>
        </div>
    </div>
    
    <div class="status" id="statusMessage"></div>
    
    <script>
        let state = {
            images: [],
            mappings: {},
            defaultImage: null,
            currentImage: null,
            pressedKeys: new Set(),
            selectedMapping: null,
            pendingKey: null
        };
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            refreshImages();
            loadMappings();
            
            // Global keyboard event handlers
            document.addEventListener('keydown', handleKeyDown);
            document.addEventListener('keyrelease', handleKeyUp);
            document.addEventListener('keyup', handleKeyUp);
            
            // Prevent default on key input
            document.getElementById('keyInput').addEventListener('keydown', (e) => {
                e.preventDefault();
                state.pendingKey = e.key;
                document.getElementById('keyInput').value = e.key;
            });
        });
        
        async function refreshImages() {
            try {
                const response = await fetch('/api/images');
                const data = await response.json();
                state.images = data.images;
                
                const imageList = document.getElementById('imageList');
                const defaultSelect = document.getElementById('defaultImageSelect');
                
                imageList.innerHTML = '';
                defaultSelect.innerHTML = '<option value="">(None)</option>';
                
                data.images.forEach(img => {
                    const option1 = document.createElement('option');
                    option1.value = img;
                    option1.textContent = img;
                    imageList.appendChild(option1);
                    
                    const option2 = document.createElement('option');
                    option2.value = img;
                    option2.textContent = img;
                    defaultSelect.appendChild(option2);
                });
                
                document.getElementById('directoryPath').textContent = data.outputs_dir;
                showStatus(`Found ${data.images.length} images`);
            } catch (error) {
                console.error('Error loading images:', error);
                showStatus('Error loading images', true);
            }
        }
        
        async function loadMappings() {
            try {
                const response = await fetch('/api/mappings');
                const data = await response.json();
                state.mappings = data.mappings;
                state.defaultImage = data.default_image;
                updateMappingsDisplay();
            } catch (error) {
                console.error('Error loading mappings:', error);
            }
        }
        
        function onImageSelect() {
            const select = document.getElementById('imageList');
            if (select.selectedIndex >= 0) {
                // Image selected, ready for mapping
            }
        }
        
        function previewSelectedImage() {
            const select = document.getElementById('imageList');
            if (select.selectedIndex >= 0) {
                const imagePath = select.value;
                displayImage(imagePath);
            }
        }
        
        function displayImage(imagePath) {
            const container = document.getElementById('imageContainer');
            const infoBar = document.getElementById('infoBar');
            
            container.innerHTML = `<img src="/api/image/${imagePath}" alt="${imagePath}">`;
            infoBar.textContent = `Viewing: ${imagePath}`;
            state.currentImage = imagePath;
        }
        
        async function setDefaultImage() {
            const select = document.getElementById('defaultImageSelect');
            const image = select.value;
            
            try {
                if (image) {
                    await fetch('/api/default', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({image})
                    });
                    state.defaultImage = image;
                    showStatus(`Default set to: ${image}`);
                } else {
                    showStatus('Please select an image');
                }
            } catch (error) {
                console.error('Error setting default:', error);
                showStatus('Error setting default', true);
            }
        }
        
        async function clearDefaultImage() {
            try {
                await fetch('/api/default', {method: 'DELETE'});
                state.defaultImage = null;
                document.getElementById('defaultImageSelect').value = '';
                showStatus('Default image cleared');
            } catch (error) {
                console.error('Error clearing default:', error);
            }
        }
        
        async function mapImageToKey() {
            const imageSelect = document.getElementById('imageList');
            const keyInput = document.getElementById('keyInput');
            
            if (imageSelect.selectedIndex < 0) {
                showStatus('Please select an image', true);
                return;
            }
            
            const key = state.pendingKey || keyInput.value;
            if (!key) {
                showStatus('Please press a key', true);
                return;
            }
            
            const image = imageSelect.value;
            
            try {
                const response = await fetch('/api/mappings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key, image})
                });
                const data = await response.json();
                state.mappings = data.mappings;
                updateMappingsDisplay();
                
                keyInput.value = '';
                state.pendingKey = null;
                showStatus(`Mapped '${key}' to ${image}`);
            } catch (error) {
                console.error('Error mapping key:', error);
                showStatus('Error creating mapping', true);
            }
        }
        
        async function clearSelectedMapping() {
            if (!state.selectedMapping) {
                showStatus('Please select a mapping to clear', true);
                return;
            }
            
            try {
                const response = await fetch('/api/mappings', {
                    method: 'DELETE',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({key: state.selectedMapping})
                });
                const data = await response.json();
                state.mappings = data.mappings;
                state.selectedMapping = null;
                updateMappingsDisplay();
                showStatus('Mapping cleared');
            } catch (error) {
                console.error('Error clearing mapping:', error);
                showStatus('Error clearing mapping', true);
            }
        }
        
        function updateMappingsDisplay() {
            const list = document.getElementById('mappingsList');
            list.innerHTML = '';
            
            if (Object.keys(state.mappings).length === 0) {
                list.innerHTML = '<div class="no-display" style="padding: 20px;">No mappings yet</div>';
                return;
            }
            
            Object.entries(state.mappings).sort().forEach(([key, image]) => {
                const item = document.createElement('div');
                item.className = 'mapping-item';
                if (state.selectedMapping === key) {
                    item.classList.add('selected');
                }
                item.innerHTML = `<span class="key-badge">${key}</span>${image}`;
                item.onclick = () => {
                    state.selectedMapping = key;
                    updateMappingsDisplay();
                };
                list.appendChild(item);
            });
        }
        
        function handleKeyDown(e) {
            // Ignore if typing in input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
                return;
            }
            
            const key = e.key;
            
            // Check if key is mapped
            if (state.mappings[key] && !state.pressedKeys.has(key)) {
                e.preventDefault();
                state.pressedKeys.add(key);
                displayImage(state.mappings[key]);
            }
        }
        
        function handleKeyUp(e) {
            const key = e.key;
            
            if (state.pressedKeys.has(key)) {
                state.pressedKeys.delete(key);
                
                // If no keys pressed and we have a default, show it
                if (state.pressedKeys.size === 0 && state.defaultImage) {
                    displayImage(state.defaultImage);
                }
            }
        }
        
        function showStatus(message, isError = false) {
            const statusEl = document.getElementById('statusMessage');
            statusEl.textContent = message;
            statusEl.style.background = isError ? '#f44336' : '#4CAF50';
            statusEl.style.display = 'block';
            
            setTimeout(() => {
                statusEl.style.display = 'none';
            }, 3000);
        }
    </script>
</body>
</html>"""


def create_template_file(template_dir):
    """Create the HTML template file."""
    template_dir.mkdir(exist_ok=True)
    template_file = template_dir / 'index.html'
    with open(template_file, 'w') as f:
        f.write(HTML_TEMPLATE)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flatmap Visualiser (Web)")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="../outputs",
        help="Path to outputs directory (default: ../outputs)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the web server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    # Resolve the directory path
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / args.dir
    outputs_dir = outputs_dir.resolve()
    
    config['outputs_dir'] = outputs_dir
    
    # Create templates directory
    template_dir = script_dir / 'templates'
    create_template_file(template_dir)
    
    print("=" * 60)
    print("Flatmap Visualiser Web Server")
    print("=" * 60)
    print(f"Images directory: {outputs_dir}")
    print(f"Server starting at: http://{args.host}:{args.port}")
    print("\nOpen this URL in your web browser to use the visualiser.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 60)
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Flatmap Visualiser (Web)")
    parser.add_argument('--port', type=int, default=5000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    app.run(debug=True, host=args.host, port=args.port)
