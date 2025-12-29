# import flask
# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from PIL import Image
# import io
# import os

# app = Flask(__name__)
# CORS(app)

# # Cấu hình đường dẫn
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, 'ML_models')
# STATIC_DIR = os.path.join(BASE_DIR, 'navigation_menu')

# # Load model đã huấn luyện
# try:
#     mobilenet_model = load_model(os.path.join(MODEL_DIR, 'MobileNet_CNN.h5'))
#     resnet_model = load_model(os.path.join(MODEL_DIR, 'Resnet152.h5'))
#     print("Model load thành công")
#     print(f"   - MobileNet: {MODEL_DIR}/MobileNet_CNN.h5")
#     print(f"   - ResNet152: {MODEL_DIR}/Resnet152.h5")
# except Exception as e:
#     print(f"lỗi load model: {e}")
#     mobilenet_model = None
#     resnet_model = None


# INGREDIENT_CLASSES = [

#     'apple',
#     'banana',
#     'beetroot',
#     'bell pepper',
#     'cabbage',
#     'capsicum',
#     'carrot',
#     'cauliflower',
#     'chilli pepper',
#     'corn',
#     'cucumber',
#     'eggplant',
#     'garlic',
#     'ginger',
#     'grapes',
#     'jalepeno',
#     'kiwi',
#     'lemon',
#     'lettuce',
#     'mango',
#     'onion',
#     'orange',
#     'paprika',
#     'pear',
#     'peas',
#     'pineapple',
#     'pomegranate',
#     'potato',
#     'raddish',
#     'soy beans',
#     'spinach',
#     'sweetcorn',
#     'sweetpotato',
#     'tomato',
#     'turnip',
#     'watermelon',
# ]


# DISH_CLASSES = [
#     'Banh bao',
#     'Banh bot loc',
#     'Banh can',
#     'Banh canh',
#     'Banh chung',
#     'Banh cuon',
#     'Banh duc',
#     'Banh gio',
#     'Banh khot',
#     'Banh mi',
#     'Banh pio',
#     'Banh tet',
#     'Banh trang nuong',
#     'Banh xeo',
#     'Bun bo Hue',
#     'Bun dau mam tom',
#     'Bun mam',
#     'Bun rieu',
#     'Ca kho to',
#     'Canh chua',
#     'Cao lau',
#     'Chao long',
#     'Com tam',
#     'Goi cuon',
#     'Hu tieu',
#     'Mi quang',
#     'Nem chua',
#     'Pho',
#     'Xoi xeo',
# ]

# def preprocess_image_mobilenet(img):
#     """Tiền xử lý cho MobileNet (224x224)"""
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     # MobileNetV2 preprocessing:   [-1, 1]
#     img_array = img_array / 127.5 - 1.0
#     return img_array

# def preprocess_image_resnet(img):
#     """Tiền xử lý cho ResNet152 (300x300)"""
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img = img.resize((300, 300)) 
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     return img_array

# # Định nghĩa các route API
# @app.route('/')
# def index():
#     return send_from_directory(STATIC_DIR, 'index.html')

# @app.route('/<path:path>')
# def serve_static(path):
#     """Serve CSS, assets, và các file HTML khác"""
#     if os.path.exists(os.path.join(STATIC_DIR, path)):
#         return send_from_directory(STATIC_DIR, path)
#     if os.path.exists(os.path.join(BASE_DIR, path)):
#         return send_from_directory(BASE_DIR, path)
#     return "File không tồn tại", 404

# @app.route('/api/health')
# def health():
#     """Check API health status"""
#     return jsonify({
#         'status': 'healthy',
#         'mobilenet_loaded': mobilenet_model is not None,
#         'resnet_loaded': resnet_model is not None,
#         'ingredient_classes': len(INGREDIENT_CLASSES),
#         'dish_classes': len(DISH_CLASSES),
#         'expected': {
#             'ingredients': 36,
#             'dishes': 30
#         }
#     })

# @app.route('/api/predict', methods=['POST'])
# def predict():
#     """Main prediction endpoint"""
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'}), 400
        
#         file = request.files['image']
#         food_type = request.form.get('type', 'dish')
        
#         if file.filename == '':
#             return jsonify({'error': 'No file selected'}), 400
        
#         img_bytes = file.read()
#         img = Image.open(io.BytesIO(img_bytes))
        
#         if food_type == 'ingredient':
#             if mobilenet_model is None:
#                 return jsonify({'error': 'MobileNet model not loaded'}), 500
            
#             processed_img = preprocess_image_mobilenet(img)
#             predictions = mobilenet_model.predict(processed_img, verbose=0)
#             classes = INGREDIENT_CLASSES
#             model_name = 'MobileNetV2'
            
#         else:  # dish
#             if resnet_model is None:
#                 return jsonify({'error': 'ResNet model not loaded'}), 500
            
#             processed_img = preprocess_image_resnet(img)
#             predictions = resnet_model.predict(processed_img, verbose=0)
#             classes = DISH_CLASSES
#             model_name = 'ResNet152V2'
        
#         # Get top 3 predictions
#         top_indices = np.argsort(predictions[0])[-3:][::-1]
#         results = []
        
#         for idx in top_indices:
#             if idx < len(classes):
#                 confidence = float(predictions[0][idx] * 100)
#                 results.append({
#                     'name': classes[idx],
#                     'confidence': confidence,
#                     'index': int(idx)
#                 })
        
#         return jsonify({
#             'status': 'success',
#             'type': food_type,
#             'model': model_name,
#             'predictions': results,
#             'total_classes': len(classes)
#         })
        
#     except Exception as e:
#         import traceback
#         return jsonify({
#             'status': 'error',
#             'message': str(e),
#             'traceback': traceback.format_exc()
#         }), 500

# @app.route('/api/classes')
# def get_classes():
#     """Get available class names"""
#     return jsonify({
#         'ingredient_classes': INGREDIENT_CLASSES,
#         'dish_classes': DISH_CLASSES,
#         'counts': {
#             'ingredients': len(INGREDIENT_CLASSES),
#             'dishes': len(DISH_CLASSES)
#         }
#     })

# if __name__ == '__main__':
#     print("=" * 70)
#     print("Food Recognition API Server")
#     print("=" * 70)
#     print(f"Base Directory: {BASE_DIR}")
#     print(f"Model Directory: {MODEL_DIR}")
#     print(f"Static Directory: {STATIC_DIR}")
#     print("=" * 70)
#     print("Classes Configuration:")
#     print(f"   - Ingredients: {len(INGREDIENT_CLASSES)} classes (expected: 36)")
#     print(f"   - Dishes:      {len(DISH_CLASSES)} classes (expected: 30)")
    
#     if len(INGREDIENT_CLASSES) != 36:
#         print(f"WARNING: Ingredient classes count mismatch!")
#     if len(DISH_CLASSES) != 30:
#         print(f"WARNING: Dish classes count mismatch!")
    
#     print("=" * 70)
#     print("Server running on: http://localhost:5000")
#     print("API Endpoints:")
#     print("   - GET  /                → Main page")
#     print("   - GET  /api/health      → Health check")
#     print("   - POST /api/predict     → Image prediction")
#     print("   - GET  /api/classes     → Get class names")
#     print("=" * 70)
    
#     app.run(debug=True, host='0.0.0.0', port=5000)





# -*- coding: utf-8 -*-
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

# ============================================
# CONFIGURATION
# ============================================

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///food_recognition.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
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
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
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

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

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

# ============================================
# STATIC FILE ROUTES
# ============================================

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

# ============================================
# API ENDPOINTS
# ============================================

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
        
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Select model and predict
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
        
        # Get top 3 predictions
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
        
        # Get nutrition info for top prediction
        top_prediction_name = results[0]['name']
        nutrition_info = get_nutrition_with_fallback(top_prediction_name)
        
        # Save to database
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
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/history')
def get_history():
    """Get recent prediction history"""
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
    print("=" * 70)
    print("🚀 Food Recognition API Server - UPGRADED")
    print("=" * 70)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print("=" * 70)
    print("✅ New Features:")
    print("   - Database + History")
    print("   - Nutrition Information")
    print("   - Statistics & Analytics")
    print("=" * 70)
    print("🌐 Server: http://localhost:5000")
    print("📊 Endpoints:")
    print("   - GET  /api/health        → Health + Stats")
    print("   - POST /api/predict       → Predict + Nutrition")
    print("   - GET  /api/history       → Recent predictions")
    print("   - GET  /api/statistics    → Dashboard data")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)