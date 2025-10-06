import numpy as np
import cv2
from math import erf
import matplotlib.pyplot as plt
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

def compute_adaptive_beta(image,M,N,II,II2):
    N_var=compute_var(II,II2,M,N)
    # print("N_var=",N_var)
    hist=compute_hist(M,N,image)
    l=256
    x = np.arange(256)
    # plt.bar(x, hist)
    # plt.title("Histogram of Input Image")
    # plt.show()
    N_H=compute_entropy(hist)
    # print("N_H=",N_H)
    N_Grad=compute_Gradient(image,M,N)
    # print("N_Grad",N_Grad)
    w1=0.4
    w2=0.3
    w3=0.3
    S=w1*N_var+w2*N_H+w3*N_Grad
    # print(S)
    gamma=1
    beta_min=3
    beta_max=21
    adaptive_beta=beta_min+((beta_max-beta_min)*(S**gamma))
    adaptive_beta=np.ceil(adaptive_beta)
    if adaptive_beta%2==0:
        adaptive_beta+=1
    # print("adaptive_beta=",adaptive_beta)
    return round(adaptive_beta)
def compute_grad(image,g,M,N):
    # Convert image to float64 to prevent integer overflow in scalar operations
    img_float = image.astype(np.float64)
    G_mat = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            x1 = g[0] * img_float[i-1, j-1] if i-1 >= 0 and j-1 >= 0 else 0.0
            x2 = g[1] * img_float[i-1, j] if i-1 >= 0 else 0.0
            x3 = g[2] * img_float[i-1, j+1] if i-1 >= 0 and j+1 < N else 0.0
            x4 = g[3] * img_float[i, j-1] if j-1 >= 0 else 0.0
            x5 = g[4] * img_float[i, j]
            x6 = g[5] * img_float[i, j+1] if j+1 < N else 0.0
            x7 = g[6] * img_float[i+1, j-1] if i+1 < M and j-1 >= 0 else 0.0
            x8 = g[7] * img_float[i+1, j] if i+1 < M else 0.0
            x9 = g[8] * img_float[i+1, j+1] if i+1 < M and j+1 < N else 0.0
            G_mat[i, j] = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
    return G_mat
def compute_Gradient(image,M,N):
    sobel_x=[-1,0,1,-2,0,2,-1,0,1]
    sobel_y=[-1,-2,-1,0,0,0,1,2,1]
    G_X=compute_grad(image,sobel_x,M,N)
    G_Y=compute_grad(image,sobel_y,M,N)
    G_XY=np.zeros((M, N))
    s=0
    for i in range(M):
        for j in range(N):
            G_XY[i][j]=((G_X[i][j]**2)+(G_Y[i][j]**2))**0.5
            s=s+G_XY[i][j]
    grad=s/(M*N)
    g_ref=1442 #370000
    n_grad=grad/g_ref
    # # print(G_X)
    # # print(G_Y)
    # print(G_XY)
    # print(s)
    # print(grad)
    return n_grad
    
def compute_entropy(hist):
    l=255
    e=1e-12
    B=256
    probab_hist=compute_probabilities(hist)
    # print(probab_hist)
    # print("sum==",sum(probab_hist))
    h=0
    for i in range(len(probab_hist)):
        h=h+(probab_hist[i]*np.log2(probab_hist[i]+e))
    h=h*-1
    # print(h)
    h=h/(np.log2(B))
    h=h*1
    h=round(h,2)
    return h
    
def compute_var(II,II2,M,N):
    U=II[M-1][N-1]
    U2=II2[M-1][N-1]
    # print(U)
    # print(U2)
    U=U/(M*N)
    U2=U2/(M*N)
    var=U2-(U*U)
    # print(U)
    # print(U*U)
    # print(U2)
    # print(var)
    vref=16256.25
    Nvar=var/vref
    return Nvar
def compute_probabilities(hist):
    total=sum(hist)
    probabilities=[]
    for i in range(len(hist)):
        probabilities.append(hist[i]/total)
    return probabilities
def compute_hist(M,N,image):
    hist=[]
    l=256
    for i in range(0,l):
        hist.append(0)
    for i in range(M):
        for j in range(N):
            k = min(int(image[i][j]), 255)
            hist[k]+=1
    return hist
def perform_SPOHE(image, stratified_regions, means_and_stds, beta, M, N, L):
    output_image = [[0 for _ in range(N)] for _ in range(M)]  # Initialize to 0 instead of -1
    K = beta * beta  # Since alpha = 1, K = (beta - alpha + 1)^2 = beta^2
    for i in range(M):
        for j in range(N):
            cdf_sum = 0.0
            for k in range(K):
                region_mean = means_and_stds[k][0]
                region_std = means_and_stds[k][1]
                if region_std == 0:
                    cdf_k = 1.0 if image[i][j] >= region_mean else 0.0
                else:
                    z = (image[i][j] - region_mean) / (np.sqrt(2) * region_std)
                    cdf_k = (1 + erf(z)) / 2
                cdf_sum += cdf_k
            # Average of CDFs from all strata
            avg_cdf = cdf_sum / K
            output_image[i][j] = round((L - 1) * avg_cdf)  # map to 0–(L-1)
    
    return output_image
def compute_mean(integral_image, stratum_region, M, N):
    top, left, bottom, right = stratum_region
    A = integral_image[bottom][right]
    B = integral_image[bottom][left - 1] if left > 0 else 0
    C = integral_image[top - 1][right] if top > 0 else 0
    D = integral_image[top - 1][left - 1] if top > 0 and left > 0 else 0
    total = A - B - C + D
    area = (bottom - top + 1) * (right - left + 1)
    mean = total / area
    return mean
def compute_mean1(integral_image,stratum_region,M,N):
    m=stratum_region[2]-stratum_region[0]+1
    n=stratum_region[3]-stratum_region[1]+1
    center_i=stratum_region[0]+int(np.floor(m/2))
    center_j=stratum_region[1]+int(np.floor(n/2))
    """partA"""
    i=center_i+int(np.floor(m/2))
    j=center_j+int(np.floor(n/2))
    partA=integral_image[i][j] if ((i>=0 and i<M) and (j>=0 and j<N)) else 0
    """partB"""
    i=center_i+int(np.floor(m/2))
    j=center_j-int(np.ceil(n/2))
    partB=integral_image[i][j] if ((i>=0 and i<M) and (j>=0 and j<N)) else 0
    """partC"""
    i=center_i-int(np.ceil(m/2))
    j=center_j+int(np.floor(n/2))
    partC=integral_image[i][j] if ((i>=0 and i<M) and (j>=0 and j<N)) else 0
    """partD"""
    i=center_i-int(np.ceil(m/2))
    j=center_j-int(np.ceil(n/2))
    partD=integral_image[i][j] if ((i>=0 and i<M) and (j>=0 and j<N)) else 0
    mean=(partA-partB-partC+partD)/(m*n)
    print(mean,m*n)
    return mean
def compute_mean_and_std_for_stratums(integral_image_1,integral_image_2,stratified_regions,beta,M,N):
    means_and_stds=[]
    for i in stratified_regions:
        new=[]
        mean=compute_mean(integral_image_1,i,M,N)
        new.append(mean)
        mean2=compute_mean(integral_image_2,i,M,N)
        # std=(mean2-(mean**2))**0.5
        std = (max(0, mean2 - (mean**2))) ** 0.5
        new.append(std)
        means_and_stds.append(new)
        # print(mean,mean2,std)
    return means_and_stds
def compute_integral_image(image,M,N,P):
    integral_image = np.zeros((M, N), dtype=np.float64)  # Use float64 to avoid overflow
    for i in range(M):
        for j in range(N):
            left=integral_image[i][j-1] if j-1>=0 else 0
            above=integral_image[i-1][j] if i-1>=0 else 0
            left_and_above=integral_image[i-1][j-1] if i-1>=0 and j-1>=0 else 0
            integral_image[i][j] = image[i][j]**P + left + above - left_and_above
    return integral_image
def print_mat(image,lst):
    print()
    print()
    for i in range(lst[0],lst[2]+1):
        for j in range(lst[1],lst[3]+1):
            print(image[i][j],end=" ")
        print()
def startify_the_image(image, alpha, beta, M, N):
    alpha = 1 
    """Only alpha=1 is supported."""
    stratified_regions = []
    row_size = M // beta
    col_size = N // beta
    row_remainder = M % beta
    col_remainder = N % beta
    row_start = 0
    for i in range(beta):
        extra_row = 1 if i < row_remainder else 0
        block_height = row_size + extra_row
        row_end = row_start + block_height - 1 
        col_start = 0
        for j in range(beta):
            extra_col = 1 if j < col_remainder else 0
            block_width = col_size + extra_col
            col_end = col_start + block_width - 1  
            stratified_regions.append([row_start, col_start, row_end, col_end])
            col_start = col_end + 1
        row_start = row_end + 1
    return stratified_regions

def ASPOHE(image_path: str, save_dir: str) -> Tuple[str, List[int], List[int]]:
    """
    Process the image using ASPOHE and return the save path, original histogram, and enhanced histogram.
    
    Args:
        image_path (str): Path to the input image.
        save_dir (str): Directory to save the enhanced image.
    
    Returns:
        Tuple[str, List[int], List[int]]: (enhanced_image_path, original_hist, enhanced_hist)
    """
    """main section"""
    image_color = cv2.imread(image_path)
    if image_color is None:
        raise FileNotFoundError(f"Image not found or could not be read: {image_path}")
    
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY).astype(np.float64)  # Cast to float64 early to avoid overflow
    M, N = image.shape
    L = 255

    # Compute original histogram (cast back to uint8 for hist computation)
    original_hist = compute_hist(M, N, image.astype(np.uint8))

    # Integral images
    integral_image_1 = compute_integral_image(image, M, N, 1)
    integral_image_2 = compute_integral_image(image, M, N, 2)

    alpha = 1
    beta = compute_adaptive_beta(image, M, N, integral_image_1, integral_image_2)
    stratified_regions = startify_the_image(image, alpha, beta, M, N)

    # Compute means and stds
    means_and_stds = compute_mean_and_std_for_stratums(
        integral_image_1, integral_image_2, stratified_regions, beta, M, N
    )

    # SPOHE
    output_img_list = perform_SPOHE(image, stratified_regions, means_and_stds, beta, M, N, L)
    output_img = np.array(output_img_list, dtype=np.uint8)
    output_img = cv2.GaussianBlur(output_img, (3, 3), sigmaX=0.5)

    # Compute enhanced histogram
    enhanced_hist = compute_hist(M, N, output_img)

    image_ycrcb = cv2.cvtColor(image_color, cv2.COLOR_BGR2YCrCb)
    image_ycrcb[:, :, 0] = output_img
    final_color_img = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Save result
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(save_dir, f"{name}_ASPOHE{ext}")
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
    
    if method != 'ASPOHE':
        return jsonify({'success': False, 'error': 'Unsupported method'}), 400
    
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(upload_path):
        return jsonify({'success': False, 'error': 'File not found'}), 404
    
    try:
        save_path, original_hist, enhanced_hist = ASPOHE(upload_path, app.config['RESULTS_FOLDER'])
        output_filename = os.path.basename(save_path)
        return jsonify({
            'success': True,
            'output_filename': output_filename,
            'original_hist': original_hist,
            'enhanced_hist': enhanced_hist
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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