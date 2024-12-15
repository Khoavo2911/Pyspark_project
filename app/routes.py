from flask import Blueprint, current_app, request, render_template, jsonify, send_file
from .models.pyspark_model import ModelPredictor
import pandas as pd
import os
import io

main = Blueprint('main', __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__) )
model_path = os.path.join(BASE_DIR, 'models', 'best_model')
model_predictor = ModelPredictor(model_path)

@main.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input data from form
            input_data = {
                'BMI': [float(request.form['BMI'])],
                'Smoking': [float(request.form['Smoking'])],
                'Stroke': [float(request.form['Stroke'])],
                'PhysicalHealth': [float(request.form['PhysicalHealth'])],
                'DiffWalking': [float(request.form['DiffWalking'])],
                'Sex': [float(request.form['Sex'])],
                'Diabetic': [float(request.form['Diabetic'])],
                'PhysicalActivity': [float(request.form['PhysicalActivity'])],
                'GenHealth': [float(request.form['GenHealth'])],   
            }
            
            # Create DataFrame
            input_df = pd.DataFrame(input_data)

            # Get prediction
            prediction = model_predictor.predict(input_df)

            # Extract prediction result
            result = prediction['prediction'].values[0]
            if result == 0.0:
                probability = prediction['probability'].values[0][0]
            else:
                probability = prediction['probability'].values[0][1]

            # Trả về JSON để JavaScript xử lý
            return jsonify({
                'result': result,
                'probability': probability
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    # Render template cho GET request
    return render_template('index.html')

@main.route('/predict_file', methods=['POST'])
def predict_file():
    try:
        # Kiểm tra xem file có được gửi không
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Không có file được tải lên'
            }), 400
        
        file = request.files['file']
        
        # Kiểm tra tên file
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Không có file được chọn'
            }), 400
        
        # Kiểm tra định dạng file
        if not file.filename.lower().endswith(('.csv', '.xlsx')):
            return jsonify({
                'success': False,
                'message': 'Chỉ hỗ trợ file CSV và Excel'
            }), 400
        
        # Đọc file
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Bỏ qua dòng đầu tiên (nếu có tên cột)
        if df.columns[0].lower() in ['bmi', 'smoking', 'stroke', 'physicalhealth', 'diffwalking', 'sex', 'diabetic', 'physicalactivity', 'genhealth']:
            df = df.iloc[1:].reset_index(drop=True)
            
        # Kiểm tra các cột bắt buộc
        required_columns = ['BMI', 'Smoking', 'Stroke', 'PhysicalHealth', 
                            'DiffWalking', 'Sex', 'Diabetic', 
                            'PhysicalActivity', 'GenHealth']
        
        # Kiểm tra xem tất cả các cột cần thiết có trong file không
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({
                'success': False,
                'message': f'Thiếu các cột bắt buộc: {", ".join(missing_columns)}'
            }), 400
        
        # Chọn các cột cần thiết
        input_df = df[required_columns]
        
        # Thực hiện dự đoán
        convert_df = model_predictor.convert_columns(input_df)
        print(convert_df.head(4))
        prediction = model_predictor.predict(convert_df)
        
        # Tạo DataFrame kết quả
        results_df = df.copy()
        results_df['Prediction'] = ["Yes" if p == 1 else "No" for p in prediction['prediction']]
        
        # Tạo tên file mới
        original_filename = file.filename
        filename_without_ext, ext = os.path.splitext(original_filename)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        new_filename = f"{filename_without_ext}_{timestamp}_predicted{ext}"

        # Lưu file Excel
        output_path = os.path.join(os.path.dirname(__file__), 'static', 'UPLOAD_FOLDER', new_filename)
        results_df.to_excel(output_path, index=False, sheet_name='Predictions')
        print('đã lưu ',output_path)
        return jsonify({
            'success': True,
            'message': 'Dự đoán thành công',
            'filename': new_filename,
            'preview_data': results_df.head(10).to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@main.route('/download_predicted_file/<filename>')
def download_predicted_file(filename):
    try:
        upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'UPLOAD_FOLDER')
        file_path = os.path.join(upload_folder, filename)
        
        if os.path.exists(file_path):
            return send_file(
                file_path,
                as_attachment=True,
                download_name=os.path.basename(filename)
            )
        else:
            return jsonify({'success': False, 'message': 'File không tồn tại'}), 404
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500