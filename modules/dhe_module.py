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
def compute_HE(l,sub_hist,lower_allowed_GL,upper_allowed_GL):
    cumm_hist=[sub_hist[0]]
    for i in range(1,len(sub_hist)):
        cumm_hist.append(cumm_hist[i-1]+sub_hist[i])
    new_GL_values=[]
    total=cumm_hist[-1]
    if total == 0:
        default_val = (lower_allowed_GL + upper_allowed_GL) // 2
        new_GL_values = [default_val] * len(sub_hist)
        return new_GL_values
    for i in range(len(cumm_hist)):
        k=round((cumm_hist[i]/total)*(upper_allowed_GL-lower_allowed_GL))
        k=k+lower_allowed_GL
        k=max(min(k, upper_allowed_GL), lower_allowed_GL)
        new_GL_values.append(k)
    return new_GL_values
def Histogram_equalization(hist_smooth,split_points,allowed_GL,l,M,N):
    new_GL_values=[]
    sub_hist=hist_smooth[split_points[0]:split_points[1]+1]
    lower_allowed_GL=split_points[0]
    upper_allowed_GL=lower_allowed_GL+allowed_GL[0]
    new_GL_values+=compute_HE(l,sub_hist,lower_allowed_GL,upper_allowed_GL)
    for i in range(1,len(split_points)-1):
        sub_hist=hist_smooth[split_points[i]+1:split_points[i+1]+1]
        lower_allowed_GL=sum(round(g) for g in allowed_GL[:i])
        upper_allowed_GL=lower_allowed_GL+allowed_GL[i]
        # if len(new_GL_values) < 256:
        #     new_GL_values += [new_GL_values[-1]] * (256 - len(new_GL_values))
        new_GL_values+=compute_HE(l,sub_hist,lower_allowed_GL,upper_allowed_GL)
    return new_GL_values
    
def calculate_range_function(factors_list,l):
    # print(factors_list,l)
    total=sum(factors_list)
    # print(total)
    range_list=[]
    for i in range(len(factors_list)):
        range_list.append(((factors_list[i]/total)*(l-1)))
    return range_list
def factors(loc_min_list,span_list,hist_smooth):
    x=2
    factors_list=[]
    l=hist_smooth[loc_min_list[0]:loc_min_list[1]+1]
    s = sum(l)
    if s == 0:
        s = 1
    factors_list.append(round(span_list[0]*((np.log(s))**x),0))
    # factors_list.append(round(span_list[0]*((np.log(sum(l)))**x),0))
    for i in range(1,len(loc_min_list)-1):
        l=hist_smooth[loc_min_list[i]+1:loc_min_list[i+1]+1]
        s = sum(l)
        if s == 0:
            s = 1
        factors_list.append(round(span_list[i]*((np.log(s))**x),0))
        # factors_list.append(round(span_list[i]*((np.log(sum(l))**x)),0))
    return factors_list
def span(loc_min_list):
    span_list=[]
    for i in range(1,len(loc_min_list)):
        span_list.append(loc_min_list[i]-loc_min_list[i-1])
    return span_list
def subdivide_using_mean_std(hist_smooth,loc_min_list,start,end):
    if end!=loc_min_list[-1]:
        l=hist_smooth[start:end+1]
    else:
        l=hist_smooth[start:]
    # print(l)
    mean=np.mean(l)
    std=np.std(l)
    comp1=mean-std
    comp2=mean+std
    total=np.sum(l)
    # print(comp1)
    # print(comp2)
    percentage=0
    for i in l:
        if i >comp1 and i<comp2:
            percentage+=i
    if total!=0:
        percentage=(percentage/total)*100
    # print(percentage)
    if percentage<68.3:
        if start < int(comp1) < end:
            region1 = int(round(comp1))
            # print(region1)
            if region1 not in loc_min_list:
                loc_min_list.append(region1)
                loc_min_list.sort()
                subdivide_using_mean_std(hist_smooth, loc_min_list, start, region1)
        
        if start < int(comp2) < end:
            region3 = int(round(comp2))
            # print(region3)
            if region3 not in loc_min_list:
                loc_min_list.append(region3)
                loc_min_list.sort()
                subdivide_using_mean_std(hist_smooth, loc_min_list, region3, end)
    # print()
def smooth(hist):
    smooth_hist=[]
    for i in range(len(hist)):
        if i-1<0:
            a=0
        else:
            a=hist[i-1]
        b=hist[i]
        if i+1>len(hist)-1:
            c=0
        else:
            c=hist[i+1]
        smooth_hist.append(round((a+b+c)/3,0))
    return smooth_hist
def local_minima(hist):
    m=[]
    m.append(0)
    for i in range(1,len(hist)-1):
        if hist[i-1]>hist[i] and hist[i]<hist[i+1]:
            m.append(i)
    m.append(len(hist)-1)
    return m
def divide(hist_smooth):
    loc_min_list=local_minima(hist_smooth)
    for i in range(len(loc_min_list)-1):
        if i==0:
            start=loc_min_list[0]
            end=loc_min_list[1]
        else:
            start=loc_min_list[i]+1
            end=loc_min_list[i+1]
        subdivide_using_mean_std(hist_smooth,loc_min_list,start,end)
    return loc_min_list
def allocate_Gray_Levels(split_points,hist_smooth,l):
    span_list=span(split_points)
    factors_list=factors(split_points,span_list,hist_smooth)
    range_list=calculate_range_function(factors_list,l)
    return range_list
def contrast_stretch(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val > min_val:
        return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return img

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

def DHE(image_path: str, save_dir: str) -> Tuple[str, List[int], List[int]]:
    """
    Process the image using DHE and return the save path, original histogram, and enhanced histogram.
    
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the enhanced image.
    
    Returns:
        Tuple[str, List[int], List[int]]: (enhanced_image_path, original_hist, enhanced_hist)
    """
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")

    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    M, N = image_gray.shape

    # Compute original histogram on initial grayscale
    original_hist = compute_hist(M, N, image_gray)

    l = 256
    x = np.arange(256)

    # preprocessing + histogram equalization pipeline
    image_gray = contrast_stretch(image_gray)
    hist = compute_hist(M, N, image_gray)

    hist_smooth = smooth(hist)
    split_points = divide(hist_smooth)
    allowed_GL = allocate_Gray_Levels(split_points, hist_smooth, l)
    new_GL_values = Histogram_equalization(hist_smooth, split_points, allowed_GL, l, M, N)
    output_img_list = map_GL_values(image_gray, new_GL_values)
    output_img = np.array(output_img_list)

    # Compute enhanced histogram on output_img
    enhanced_hist = compute_hist(M, N, output_img)

    # replace luminance channel with enhanced image
    image_ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    image_ycrcb[:, :, 0] = output_img
    final_color_img = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

    # save output with _DHE suffix
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(save_dir, f"{name}_DHE{ext}")
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
    
    if method == 'DHE':
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(upload_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        try:
            save_path, original_hist, enhanced_hist = DHE(upload_path, app.config['RESULTS_FOLDER'])
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