import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from typing import Tuple, List

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

def map_GL_values(image, new_GL_values):
    output_img = []
    for i in range(len(image)):
        p = []
        for j in range(len(image[0])):
            p.append(new_GL_values[image[i][j]])
        output_img.append(p)
    return output_img
def compute_hist(M,N,image):
    hist=[]
    for i in range(0,256):
        hist.append(0)
    for i in range(M):
        for j in range(N):
            if int(image[i][j])<256:
                k=int(image[i][j])
            else:
                k=255
            hist[k]+=1
    return hist
def compute_cummulative_probabs(probabs):
    cum_probabs=[probabs[0]]
    for i in range(1,len(probabs)):
        cum_probabs.append(cum_probabs[i-1]+probabs[i])
    return cum_probabs
def compute_probabilities(hist):
    total=sum(hist)
    probabilities=[]
    for i in range(len(hist)):
        probabilities.append(hist[i]/total)
    return probabilities
def recursive_segmentation_using_median(split_points,recursion_level,cum_probabs,start,end):
    if recursion_level!=0:
        m_l=cum_probabs[start]
        m_u=cum_probabs[end]
        min_D=start
        min_D_value=abs(cum_probabs[start]-((m_l+m_u)/2))
        for i in range(start+1,end+1):
            current_D_value=abs(cum_probabs[i]-((m_l+m_u)/2))
            if current_D_value<min_D_value:
                min_D_value=current_D_value
                min_D=i
        if min_D not in split_points:
            split_points.append(min_D)
            split_points.sort()
            recursive_segmentation_using_median(split_points,recursion_level-1,cum_probabs,start,min_D)
            recursive_segmentation_using_median(split_points,recursion_level-1,cum_probabs,min_D,end)
def recursive_segmentation_using_mean(split_points,recursion_level,probabs,start,end):
    if recursion_level!=0:
        mean=0
        for i in range(start,end+1):
            mean=mean+(i*probabs[i])
        total=sum(probabs[start:end+1])
        mean=mean/total
        mean=round(mean)
        if mean not in split_points:
            split_points.append(mean)
            split_points.sort()
            recursive_segmentation_using_mean(split_points,recursion_level-1,probabs,start,mean)
            recursive_segmentation_using_mean(split_points,recursion_level-1,probabs,mean+1,end)
def calculate_weighted_probabs(probabs,split_points,l):
    p_max_value=max(probabs)
    p_min_value=min(probabs)
    p_max=probabs.index(p_max_value)
    p_min=probabs.index(p_min_value)
    alpha=[]
    k=probabs[split_points[0]:split_points[1]+1]
    alpha.append(sum(k))
    for i in range(1,len(split_points)-1):
        k=probabs[split_points[i]+1:split_points[i+1]+1]
        alpha.append(sum(k))
    x_G=(0+(l-1))/2
    x_M=0
    x_max=l-1
    x_min=0
    for i in range(len(probabs)):
            x_M=x_M+(i*probabs[i])
    # beta=((p_max)*abs(x_M-x_G))/(x_max-x_min)
    # print(beta)
    beta=0
    weighted_probabs=[]
    # print("probabs =", probabs)
    # print("p_min =", p_min, "p_max =", p_max)
    # print("alpha =", alpha)
    # print("beta =", beta)
    for i in range(len(probabs)):
        for j in range(1,len(split_points)):
            if i<=split_points[j]:
                alpha_value=alpha[j-1]
                break
        if p_max!=p_min:
            normalized=((probabs[i]-p_min)/(p_max-p_min))
        else:
            normalized=0
        weighted_probabs.append(alpha_value*(1-normalized)+beta)# Inverse Proportional Weighting
        # print(normalized)
        # normalized = max(0, min(normalized, 1))
        # print(normalized**alpha_value)
        # weighted_probabs.append(p_max*(normalized**alpha_value)+beta)
    # print("weighted_probabs=",weighted_probabs)
    # print(sum(weighted_probabs))
    total=sum(weighted_probabs)
    # print(total)
    modified_weighted_probabs=[]
    for i in range(len(probabs)):
        modified_weighted_probabs.append(weighted_probabs[i]/total)
    # print("modified_weighted_probabs=",modified_weighted_probabs)
    # print(sum(modified_weighted_probabs))
    return modified_weighted_probabs
def compute_new_GL_values(cum_weighted_probabs,x0,xl):
    new_GL_values=[]
    for i in range(len(cum_weighted_probabs)):
        new_GL_values.append(round(x0+(xl-x0)*cum_weighted_probabs[i]))
    return new_GL_values
def contrast_stretch(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img

def RSWHE(image_path: str, save_dir: str) -> Tuple[str, List[int], List[int]]:
    """
    Process the image using RSWHE and return the save path, original histogram, and enhanced histogram.
    
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the enhanced image.
    
    Returns:
        Tuple[str, List[int], List[int]]: (enhanced_image_path, original_hist, enhanced_hist)
    """
    """main section for RSWHE-based enhancement"""
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    M, N = image_gray.shape

    # Compute original histogram on initial grayscale
    original_hist = compute_hist(M, N, image_gray)

    l = 256
    x = np.arange(256)

    # Contrast stretch
    image_gray = contrast_stretch(image_gray)

    # Histogram
    hist = compute_hist(M, N, image_gray)

    # Histogram segmentation
    mode = 1
    recursion_level = 2
    split_points = [0, l - 1]
    probabs = compute_probabilities(hist)
    cum_probabs = compute_cummulative_probabs(probabs)
    if mode == 1:
        recursive_segmentation_using_mean(split_points, recursion_level, probabs, 0, l - 1)
    else:
        recursive_segmentation_using_median(split_points, recursion_level, cum_probabs, 0, l - 1)

    # Histogram weighting
    weighted_probabs = calculate_weighted_probabs(probabs, split_points, l)

    # Histogram Equalization
    cum_weighted_probabs = compute_cummulative_probabs(weighted_probabs)
    x0, xl = 0, l - 1
    new_GL_values = compute_new_GL_values(cum_weighted_probabs, x0, xl)
    output_img_list = map_GL_values(image_gray, new_GL_values)
    output_img = np.array(output_img_list)

    # Compute enhanced histogram on output_img
    enhanced_hist = compute_hist(M, N, output_img)

    # Reconstruct color image
    image_ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    image_ycrcb[:, :, 0] = output_img
    final_color_img = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Save result in save_dir with same name + "_RSWHE"
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(save_dir, f"{name}_RSWHE{ext}")
    cv2.imwrite(save_path, final_color_img)

    print(f"Processed and saved: {save_path}")
    return save_path, original_hist, enhanced_hist

# Flask Routes
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'success': True, 'filename': filename})

@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    filename = data.get('filename')
    method = data.get('method')
    
    if not filename or not method:
        return jsonify({'success': False, 'error': 'Missing filename or method'}), 400
    
    if method == 'RSWHE':
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(upload_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        try:
            save_path, original_hist, enhanced_hist = RSWHE(upload_path, app.config['RESULTS_FOLDER'])
            output_filename = os.path.basename(save_path)
            return jsonify({
                'success': True,
                'output_filename': output_filename,
                'original_hist': original_hist,
                'enhanced_hist': enhanced_hist
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'Unsupported method'}), 400

@app.route('/get_image/<folder>/<filename>')
def get_image(folder, filename):
    if folder not in ['uploads', 'results']:
        return 'Invalid folder', 400
    filepath = os.path.join(app.config[folder.upper() + '_FOLDER'], filename)
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