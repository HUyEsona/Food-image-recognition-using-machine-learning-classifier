from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import os

from nutrition import get_nutrition_with_fallback

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ML_models')
STATIC_DIR = os.path.join(BASE_DIR, 'navigation_menu')
CSS_DIR = os.path.join(BASE_DIR, 'CSS')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
JS_DIR = os.path.join(BASE_DIR, 'JS')

try:
    mobilenet_model = load_model(os.path.join(MODEL_DIR, 'MobileNet_CNN.h5'))
    resnet_model = load_model(os.path.join(MODEL_DIR, 'Resnet50.h5'))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    mobilenet_model = None
    resnet_model = None

INGREDIENT_CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum',
    'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce',
    'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip', 'watermelon',
]

DISH_CLASSES = [
    'baba_nau_chuoi_dau', 'banh tet', 'banh_bao', 'banh_beo', 'banh_bo',
      'banh_bot_loc', 'banh_can', 'banh_canh', 'banh_chung', 'banh_cong',
        'banh_cuon', 'banh_da_cua', 'banh_da_lon', 'banh_duc', 'banh_gai', 'banh_giay',
          'banh_gio', 'banh_khot', 'banh_la', 'banh_mi', 'banh_pia', 'banh_tai_heo', 'banh_tieu',
            'banh_tom_ho_tay', 'banh_trang_nuong', 'banh_troi_nuoc', 'banh_trung_thu', 'banh_u',
              'banh_xeo', 'bo_kho', 'bo_luc_lac', 'bo_ne', 'bo_nuong_la_lot', 'bun_bo_hue', 'bun_cha',
                'bun_cha_ca', 'bun_dau_mam_tom', 'bun_mam', 'bun_rieu', 'bun_thit_nuong', 'ca_kho_to', 'ca_loc_nuong',
                  'ca_muoi_xoi', 'ca_ri_ga', 'ca_sot_ca_chua', 'canh_bi_do', 'canh_chua', 'canh_cua', 'canh_kho_hoa',
                    'canh_khoai_tim', 'cao_lau', 'cha_ca_la_vong', 'cha_com', 'cha_lui', 'chao_long', 'chao_vit',
                      'com_chay_cha_bong', 'com_chien', 'com_ga_xoi_mo', 'com_lam', 'com_rang_dua_bo', 'com_tam',
                        'cua_hap_bia', 'cut_lon_xao_me', 'ga_chien_nuoc_mam', 'ga_hap_la_chanh', 'goi_ca_chich', 
                        'goi_cuon', 'hu_tieu', 'khau_nhuc', 'kho_muc_nuong', 'kho_quet', 'lap_xuong',
                          'luon_om_chuoi_dau', 'luon_xao_xa_ot', 'mam_chung', 'mam_tep_chung_thit', 'mi_quang',
                            'mi_xao_gion', 'muc_nhoi_thit', 'nam_pia', 'nem_chua', 'nem_nuong_nha_trang', 'nui_xao',
                              'oc_buou_hap', 'oc_huong_xao', 'oc_len_xao_dua', 'pho', 'rau_muong_xao', 'sup_cua',
                                'tau_hu_nhoi_thit', 'tau_hu_non', 'thit_dong', 'thit_kho_tau', 'thit_trau_gac_bep',
                                  'tiet_canh', 'trung_vit_lon', 'xoi_gac', 'xoi_nep_than', 'xoi_xeo'
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

@app.route('/CSS/<path:filename>')
def serve_css(filename):
    return send_from_directory(CSS_DIR, filename)

@app.route('/JS/<path:filename>')
def serve_js(filename):
    return send_from_directory(JS_DIR, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/navigation_menu/<path:filename>')
def serve_navigation(filename):
    return send_from_directory(STATIC_DIR, filename)



@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'mobilenet_loaded': mobilenet_model is not None,
        'resnet_loaded': resnet_model is not None,
        'ingredient_classes': len(INGREDIENT_CLASSES),
        'dish_classes': len(DISH_CLASSES),
    })

@app.route('/api/predict', methods=['POST'])
def predict():
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
        else:
            if resnet_model is None:
                return jsonify({'error': 'ResNet model not loaded'}), 500
            
            processed_img = preprocess_image_resnet(img)
            predictions = resnet_model.predict(processed_img, verbose=0)
            classes = DISH_CLASSES
            model_name = 'ResNet50'
        
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
        
        top_prediction_name = results[0]['name']
        nutrition_info = get_nutrition_with_fallback(top_prediction_name)
        
        return jsonify({
            'status': 'success',
            'type': food_type,
            'model': model_name,
            'predictions': results,
            'nutrition': nutrition_info,
            'total_classes': len(classes)
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/classes')
def get_classes():
    return jsonify({
        'ingredient_classes': INGREDIENT_CLASSES,
        'dish_classes': DISH_CLASSES,
        'counts': {
            'ingredients': len(INGREDIENT_CLASSES),
            'dishes': len(DISH_CLASSES)
        }
    })

if __name__ == '__main__':
    print("Đang chạy Food Recognition API")
    print("Server: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)