
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

# Import database và nutrition modules
from database import db, init_db, save_prediction, get_recent_predictions, get_statistics, get_overall_statistics
from nutrition import get_nutrition_with_fallback

app = Flask(__name__)
CORS(app)



# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food_recognition.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# khởi tạo database
init_db(app)

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_models')
STATIC_DIR = os.path.join(BASE_DIR, 'navigation_menu')
CSS_DIR = os.path.join(BASE_DIR, 'CSS')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Load models
try:
    mobilenet_model = load_model(os.path.join(MODEL_DIR, 'MobileNet_CNN.h5'))
    resnet_model = load_model(os.path.join(MODEL_DIR, 'Resnet152.h5'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    mobilenet_model = None
    resnet_model = None

# Class names
INGREDIENT_CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon',
]

DISH_CLASSES = [
    'Banh bao', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung',
    'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi',
    'Banh pio', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue',
    'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Ca kho to', 'Canh chua',
    'Cao lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu',
    'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo',
]


def preprocess_image_mobilenet(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 127.5 - 1.0
    return img_array

def preprocess_image_resnet(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array



@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/__CSS__/<path:filename>')
def serve_css(filename):
    return send_from_directory(CSS_DIR, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/navigation_menu/<path:filename>')
def serve_navigation(filename):
    return send_from_directory(STATIC_DIR, filename)



@app.route('/api/health')
def health():
    """Check API health status"""
    overall_stats = get_overall_statistics()
    
    return jsonify({
        'status': 'healthy',
        'mobilenet_loaded': mobilenet_model is not None,
        'resnet_loaded': resnet_model is not None,
        'database_connected': True,
        'total_predictions': overall_stats['total_predictions'],
        'ingredient_classes': len(INGREDIENT_CLASSES),
        'dish_classes': len(DISH_CLASSES),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with nutrition info"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        food_type = request.form.get('type', 'dish')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # đọc image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # chọn model and dự đoán
        if food_type == 'ingredient':
            if mobilenet_model is None:
                return jsonify({'error': 'MobileNet model not loaded'}), 500
            
            processed_img = preprocess_image_mobilenet(img)
            predictions = mobilenet_model.predict(processed_img, verbose=0)
            classes = INGREDIENT_CLASSES
            model_name = 'MobileNetV2'
        else:
            if resnet_model is None:
                return jsonify({'error': 'ResNet model not loaded'}), 500
            
            processed_img = preprocess_image_resnet(img)
            predictions = resnet_model.predict(processed_img, verbose=0)
            classes = DISH_CLASSES
            model_name = 'ResNet152V2'
        
        # lấy top 3 dự đoán
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        results = []
        
        for idx in top_indices:
            if idx < len(classes):
                confidence = float(predictions[0][idx] * 100)
                results.append({
                    'name': classes[idx],
                    'confidence': confidence,
                    'index': int(idx)
                })
        
        # lấy thông tin nutrition từ top dự đoán
        top_prediction_name = results[0]['name']
        nutrition_info = get_nutrition_with_fallback(top_prediction_name)
        
        # lưu vào database
        user_ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        
        prediction_id = save_prediction(
            image_name=file.filename,
            food_type=food_type,
            model_name=model_name,
            predictions=results,
            user_ip=user_ip,
            user_agent=user_agent[:255]  # Limit length
        )
        
        return jsonify({
            'status': 'success',
            'prediction_id': prediction_id,
            'type': food_type,
            'model': model_name,
            'predictions': results,
            'nutrition': nutrition_info,
            'total_classes': len(classes)
        })
        
    except Exception as e:
        import traceback
        print(f" Error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/history')
def get_history():
    limit = request.args.get('limit', 10, type=int)
    
    try:
        predictions = get_recent_predictions(limit=limit)
        
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'history': [p.to_dict() for p in predictions]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/statistics')
def statistics():
    """Get statistics for dashboard"""
    days = request.args.get('days', 7, type=int)
    
    try:
        # Lấy thống kê tổng hợp
        overall = get_overall_statistics()
        
        # Lấy thống kê theo ngày
        daily_stats = get_statistics(days=days)
        
        return jsonify({
            'status': 'success',
            'overall': overall,
            'daily': [s.to_dict() for s in daily_stats]
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/classes')
def get_classes():
    """Get available class names"""
    return jsonify({
        'ingredient_classes': INGREDIENT_CLASSES,
        'dish_classes': DISH_CLASSES,
        'counts': {
            'ingredients': len(INGREDIENT_CLASSES),
            'dishes': len(DISH_CLASSES)
        }
    })

if __name__ == '__main__':
    print("Starting Food Recognition API...")
    print(f"Using database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("Server: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)