import flask
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_models')
STATIC_DIR = os.path.join(BASE_DIR, 'navigation_menu')

# Load model đã huấn luyện
try:
    mobilenet_model = load_model(os.path.join(MODEL_DIR, 'MobileNet_CNN.h5'))
    resnet_model = load_model(os.path.join(MODEL_DIR, 'Resnet152.h5'))
    print("Model load thành công")
    print(f"   - MobileNet: {MODEL_DIR}/MobileNet_CNN.h5")
    print(f"   - ResNet152: {MODEL_DIR}/Resnet152.h5")
except Exception as e:
    print(f"lỗi load model: {e}")
    mobilenet_model = None
    resnet_model = None


INGREDIENT_CLASSES = [

    'apple',
    'banana',
    'beetroot',
    'bell pepper',
    'cabbage',
    'capsicum',
    'carrot',
    'cauliflower',
    'chilli pepper',
    'corn',
    'cucumber',
    'eggplant',
    'garlic',
    'ginger',
    'grapes',
    'jalepeno',
    'kiwi',
    'lemon',
    'lettuce',
    'mango',
    'onion',
    'orange',
    'paprika',
    'pear',
    'peas',
    'pineapple',
    'pomegranate',
    'potato',
    'raddish',
    'soy beans',
    'spinach',
    'sweetcorn',
    'sweetpotato',
    'tomato',
    'turnip',
    'watermelon',
]


DISH_CLASSES = [
    'Banh bao',
    'Banh bot loc',
    'Banh can',
    'Banh canh',
    'Banh chung',
    'Banh cuon',
    'Banh duc',
    'Banh gio',
    'Banh khot',
    'Banh mi',
    'Banh pio',
    'Banh tet',
    'Banh trang nuong',
    'Banh xeo',
    'Bun bo Hue',
    'Bun dau mam tom',
    'Bun mam',
    'Bun rieu',
    'Ca kho to',
    'Canh chua',
    'Cao lau',
    'Chao long',
    'Com tam',
    'Goi cuon',
    'Hu tieu',
    'Mi quang',
    'Nem chua',
    'Pho',
    'Xoi xeo',
]

def preprocess_image_mobilenet(img):
    """Tiền xử lý cho MobileNet (224x224)"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # MobileNetV2 preprocessing:   [-1, 1]
    img_array = img_array / 127.5 - 1.0
    return img_array

def preprocess_image_resnet(img):
    """Tiền xử lý cho ResNet152 (300x300)"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((300, 300)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Định nghĩa các route API
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve CSS, assets, và các file HTML khác"""
    if os.path.exists(os.path.join(STATIC_DIR, path)):
        return send_from_directory(STATIC_DIR, path)
    if os.path.exists(os.path.join(BASE_DIR, path)):
        return send_from_directory(BASE_DIR, path)
    return "File không tồn tại", 404

@app.route('/api/health')
def health():
    """Check API health status"""
    return jsonify({
        'status': 'healthy',
        'mobilenet_loaded': mobilenet_model is not None,
        'resnet_loaded': resnet_model is not None,
        'ingredient_classes': len(INGREDIENT_CLASSES),
        'dish_classes': len(DISH_CLASSES),
        'expected': {
            'ingredients': 36,
            'dishes': 30
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        food_type = request.form.get('type', 'dish')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        if food_type == 'ingredient':
            if mobilenet_model is None:
                return jsonify({'error': 'MobileNet model not loaded'}), 500
            
            processed_img = preprocess_image_mobilenet(img)
            predictions = mobilenet_model.predict(processed_img, verbose=0)
            classes = INGREDIENT_CLASSES
            model_name = 'MobileNetV2'
            
        else:  # dish
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
        
        return jsonify({
            'status': 'success',
            'type': food_type,
            'model': model_name,
            'predictions': results,
            'total_classes': len(classes)
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
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
    print("Food Recognition API Server")
    print("=" * 70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Static Directory: {STATIC_DIR}")
    print("=" * 70)
    print("Classes Configuration:")
    print(f"   - Ingredients: {len(INGREDIENT_CLASSES)} classes (expected: 36)")
    print(f"   - Dishes:      {len(DISH_CLASSES)} classes (expected: 30)")
    
    if len(INGREDIENT_CLASSES) != 36:
        print(f"WARNING: Ingredient classes count mismatch!")
    if len(DISH_CLASSES) != 30:
        print(f"WARNING: Dish classes count mismatch!")
    
    print("=" * 70)
    print("Server running on: http://localhost:5000")
    print("API Endpoints:")
    print("   - GET  /                → Main page")
    print("   - GET  /api/health      → Health check")
    print("   - POST /api/predict     → Image prediction")
    print("   - GET  /api/classes     → Get class names")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)