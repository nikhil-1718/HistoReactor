# app.py - Full backend with enhanced logging for upload debugging
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import os
import logging

from modules.dhe_module import DHE
from modules.rswhe_module import RSWHE
from modules.aspohe_module import ASPOHE

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Increased for multiple files

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.debug("Upload endpoint hit")
    if 'file' not in request.files:
        logger.error("No 'file' key in form data")
        return jsonify({'success': False, 'error': 'No file part'}), 400

    files = request.files.getlist('file')
    logger.debug(f"Received {len(files)} file parts")

    if not files or all(f.filename == '' for f in files):
        logger.error("No files with filenames")
        return jsonify({'success': False, 'error': 'No selected files'}), 400

    filenames = []
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                filenames.append(filename)
                logger.info(f"Uploaded {filename}")
            except Exception as e:
                logger.error(f"Save error for {filename}: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500

    if filenames:
        logger.info(f"Upload success: {filenames}")
        return jsonify({'success': True, 'filenames': filenames})
    else:
        return jsonify({'success': False, 'error': 'No valid files saved'}), 400

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    filenames_list = data.get('filenames', [])
    method = data.get('method')

    if not filenames_list or not method:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    output_filenames = []
    original_hists = []
    enhanced_hists = []

    for filename in filenames_list:
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(upload_path):
            return jsonify({'success': False, 'error': f'File not found: {filename}'}), 404

        try:
            if method == 'DHE':
                save_path, orig_hist, enh_hist = DHE(upload_path, app.config['RESULTS_FOLDER'])
            elif method == 'RSWHE':
                save_path, orig_hist, enh_hist = RSWHE(upload_path, app.config['RESULTS_FOLDER'])
            elif method == 'ASPOHE':
                save_path, orig_hist, enh_hist = ASPOHE(upload_path, app.config['RESULTS_FOLDER'])
            else:
                return jsonify({'success': False, 'error': 'Invalid method'}), 400

            output_filenames.append(os.path.basename(save_path))
            original_hists.append(orig_hist or [0] * 256)  # Fallback if hist None
            enhanced_hists.append(enh_hist or [0] * 256)
            logger.info(f"Processed {filename} -> {os.path.basename(save_path)}")
        except Exception as e:
            logger.error(f"Process error for {filename}: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({
        'success': True,
        'output_filenames': output_filenames,
        'original_hists': original_hists,
        'enhanced_hists': enhanced_hists
    })

@app.route('/get_image/<folder>/<filename>')
def get_image(folder, filename):
    if folder not in ['uploads', 'results']:
        return 'Invalid folder', 400
    folder_path = app.config['UPLOAD_FOLDER'] if folder == 'uploads' else app.config['RESULTS_FOLDER']
    filepath = os.path.join(folder_path, filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    return send_file(filepath, mimetype='image/*')

@app.route('/download/<filename>')
def download_image(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)