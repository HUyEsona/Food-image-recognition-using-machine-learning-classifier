

import requests
from typing import Dict, Optional

# Database dinh dưỡng cơ bản 
NUTRITION_DATABASE = {
    # trái cây và rau củ (per 100g)
    'apple': {
        'calories': 52,
        'protein': 0.3,
        'carbs': 14,
        'fat': 0.2,
        'fiber': 2.4,
        'vitamin_c': 4.6,
        'benefits': ['Giàu chất xơ', 'Tốt cho tim mạch', 'Chứa nhiều chất chống oxi hóa'],
    },
    'banana': {
        'calories': 89,
        'protein': 1.1,
        'carbs': 23,
        'fat': 0.3,
        'fiber': 2.6,
        'potassium': 358,
        'benefits': ['Giàu kali', 'Tăng năng lượng', 'Tốt cho tiêu hóa'],
    },
    'carrot': {
        'calories': 41,
        'protein': 0.9,
        'carbs': 10,
        'fat': 0.2,
        'fiber': 2.8,
        'vitamin_a': 835,
        'benefits': ['Tốt cho mắt', 'Giàu beta-carotene', 'Tăng cường miễn dịch'],
    },
    'tomato': {
        'calories': 18,
        'protein': 0.9,
        'carbs': 3.9,
        'fat': 0.2,
        'fiber': 1.2,
        'vitamin_c': 13.7,
        'benefits': ['Giàu lycopene', 'Chống ung thư', 'Tốt cho da'],
    },
    'potato': {
        'calories': 77,
        'protein': 2,
        'carbs': 17,
        'fat': 0.1,
        'fiber': 2.1,
        'potassium': 425,
        'benefits': ['Giàu năng lượng', 'Chứa nhiều kali', 'Tốt cho tiêu hóa'],
    },
    'cucumber': {
        'calories': 15,
        'protein': 0.7,
        'carbs': 3.6,
        'fat': 0.1,
        'fiber': 0.5,
        'vitamin_k': 16.4,
        'benefits': ['Giàu nước', 'Giúp giải nhiệt', 'Tốt cho da'],
    },
    'orange': {
        'calories': 47,
        'protein': 0.9,
        'carbs': 12,
        'fat': 0.1,
        'fiber': 2.4,
        'vitamin_c': 53.2,
        'benefits': ['Giàu vitamin C', 'Tăng cường miễn dịch', 'Chống cảm cúm'],
    },
    'lettuce': {
        'calories': 15,
        'protein': 1.4,
        'carbs': 2.9,
        'fat': 0.2,
        'fiber': 1.3,
        'vitamin_k': 126,
        'benefits': ['Ít calo', 'Giàu vitamin K', 'Tốt cho xương'],
    },
    'mango': {
        'calories': 60,
        'protein': 0.8,
        'carbs': 15,
        'fat': 0.4,
        'fiber': 1.6,
        'vitamin_c': 36.4,
        'benefits': ['Giàu vitamin A', 'Tốt cho da', 'Tăng miễn dịch'],
    },
    'grapes': {
        'calories': 69,
        'protein': 0.7,
        'carbs': 18,
        'fat': 0.2,
        'fiber': 0.9,
        'benefits': ['Chống oxi hóa', 'Tốt cho tim', 'Chống lão hóa'],
    },
    'watermelon': {
        'calories': 30,
        'protein': 0.6,
        'carbs': 8,
        'fat': 0.2,
        'fiber': 0.4,
        'benefits': ['Giàu nước', 'Giải nhiệt', 'Ít calo'],
    },
    'spinach': {
        'calories': 23,
        'protein': 2.9,
        'carbs': 3.6,
        'fat': 0.4,
        'fiber': 2.2,
        'benefits': ['Giàu sắt', 'Tốt cho máu', 'Nhiều vitamin K'],
    },
    'bell pepper': {
        'calories': 31,
        'protein': 1,
        'carbs': 6,
        'fat': 0.3,
        'fiber': 2.1,
        'benefits': ['Giàu vitamin C', 'Màu sắc đẹp', 'Chống oxi hóa'],
    },
    'onion': {
        'calories': 40,
        'protein': 1.1,
        'carbs': 9,
        'fat': 0.1,
        'fiber': 1.7,
        'benefits': ['Kháng khuẩn', 'Tốt cho tim', 'Chống viêm'],
    },
    'garlic': {
        'calories': 149,
        'protein': 6.4,
        'carbs': 33,
        'fat': 0.5,
        'fiber': 2.1,
        'benefits': ['Kháng sinh tự nhiên', 'Tăng miễn dịch', 'Giảm huyết áp'],
    },
    'eggplant': {
        'calories': 25,
        'protein': 1,
        'carbs': 6,
        'fat': 0.2,
        'fiber': 3,
        'benefits': ['Ít calo', 'Giàu chất xơ', 'Chống oxi hóa'],
    },
    'cabbage': {
        'calories': 25,
        'protein': 1.3,
        'carbs': 6,
        'fat': 0.1,
        'fiber': 2.5,
        'benefits': ['Tốt cho tiêu hóa', 'Chống ung thư', 'Giàu vitamin C'],
    },
    'cauliflower': {
        'calories': 25,
        'protein': 1.9,
        'carbs': 5,
        'fat': 0.3,
        'fiber': 2,
        'benefits': ['Ít calo', 'Giàu vitamin C', 'Chống viêm'],
    },
    'corn': {
        'calories': 86,
        'protein': 3.3,
        'carbs': 19,
        'fat': 1.4,
        'fiber': 2.7,
        'benefits': ['Giàu năng lượng', 'Chứa chất xơ', 'Tốt cho tiêu hóa'],
    },
    'peas': {
        'calories': 81,
        'protein': 5.4,
        'carbs': 14,
        'fat': 0.4,
        'fiber': 5.7,
        'benefits': ['Giàu protein thực vật', 'Nhiều chất xơ', 'Tốt cho tiêu hóa'],
    },
    'pineapple': {
        'calories': 50,
        'protein': 0.5,
        'carbs': 13,
        'fat': 0.1,
        'fiber': 1.4,
        'benefits': ['Giàu enzyme', 'Hỗ trợ tiêu hóa', 'Chống viêm'],
    },
    'lemon': {
        'calories': 29,
        'protein': 1.1,
        'carbs': 9,
        'fat': 0.3,
        'fiber': 2.8,
        'vitamin_c': 53,
        'benefits': ['Giàu vitamin C', 'Giải độc', 'Tăng cường miễn dịch'],
    },
    'ginger': {
        'calories': 80,
        'protein': 1.8,
        'carbs': 18,
        'fat': 0.8,
        'fiber': 2,
        'benefits': ['Chống buồn nôn', 'Chống viêm', 'Tốt cho tiêu hóa'],
    },
    'beetroot': {
        'calories': 43,
        'protein': 1.6,
        'carbs': 10,
        'fat': 0.2,
        'fiber': 2.8,
        'benefits': ['Tăng sức bền', 'Tốt cho gan', 'Giàu chất chống oxi hóa'],
    },
    'kiwi': {
        'calories': 61,
        'protein': 1.1,
        'carbs': 15,
        'fat': 0.5,
        'fiber': 3,
        'vitamin_c': 93,
        'benefits': ['Rất giàu vitamin C', 'Hỗ trợ miễn dịch', 'Tốt cho tiêu hóa'],
    },
    'pear': {
        'calories': 57,
        'protein': 0.4,
        'carbs': 15,
        'fat': 0.1,
        'fiber': 3.1,
        'benefits': ['Giàu chất xơ', 'Tốt cho tim', 'Ít calo'],
    },
    'pomegranate': {
        'calories': 83,
        'protein': 1.7,
        'carbs': 19,
        'fat': 1.2,
        'fiber': 4,
        'benefits': ['Chống oxi hóa mạnh', 'Tốt cho tim', 'Chống viêm'],
    },
    
    # Món ăn Việt Nam (mỗi khẩu phần ~300-400g)
    'Pho': {
        'calories': 350,
        'protein': 15,
        'carbs': 50,
        'fat': 8,
        'fiber': 3,
        'sodium': 1200,
        'benefits': ['Đầy đủ dinh dưỡng', 'Giàu protein', 'Nước dùng bổ dưỡng'],
    },
    'Banh mi': {
        'calories': 400,
        'protein': 18,
        'carbs': 45,
        'fat': 15,
        'fiber': 2,
        'sodium': 800,
        'benefits': ['Tiện lợi', 'Đa dạng dinh dưỡng', 'Ngon miệng'],
    },
    'Com tam': {
        'calories': 450,
        'protein': 20,
        'carbs': 60,
        'fat': 12,
        'fiber': 2,
        'sodium': 900,
        'benefits': ['Nhiều năng lượng', 'Giàu protein', 'Đầy bụng'],
    },
    'Bun bo Hue': {
        'calories': 420,
        'protein': 22,
        'carbs': 48,
        'fat': 14,
        'fiber': 3,
        'sodium': 1500,
        'benefits': ['Cay nóng', 'Giàu protein', 'Kích thích tiêu hóa'],
    },
    'Goi cuon': {
        'calories': 150,
        'protein': 8,
        'carbs': 20,
        'fat': 4,
        'fiber': 2,
        'sodium': 400,
        'benefits': ['Ít calo', 'Tươi mát', 'Nhiều rau sống'],
    },
    'Banh xeo': {
        'calories': 320,
        'protein': 12,
        'carbs': 35,
        'fat': 14,
        'fiber': 2,
        'sodium': 700,
        'benefits': ['Giòn ngon', 'Nhiều rau', 'Protein từ tôm thịt'],
    },
    'Banh cuon': {
        'calories': 200,
        'protein': 10,
        'carbs': 30,
        'fat': 5,
        'fiber': 1,
        'sodium': 600,
        'benefits': ['Nhẹ nhàng', 'Dễ tiêu hóa', 'Ít dầu mỡ'],
    },
    'Hu tieu': {
        'calories': 380,
        'protein': 16,
        'carbs': 52,
        'fat': 10,
        'fiber': 2,
        'sodium': 1100,
        'benefits': ['Nước dùng ngọt', 'Đa dạng topping', 'Đầy đủ dinh dưỡng'],
    },
    'Banh bao': {
        'calories': 250,
        'protein': 8,
        'carbs': 40,
        'fat': 6,
        'fiber': 1,
        'sodium': 500,
        'benefits': ['Tiện lợi', 'Ấm nóng', 'Đầy bụng'],
    },
    'Cao lau': {
        'calories': 400,
        'protein': 18,
        'carbs': 55,
        'fat': 12,
        'fiber': 3,
        'sodium': 900,
        'benefits': ['Đặc sản Hội An', 'Độc đáo', 'Giàu hương vị'],
    },
}

def get_nutrition_info(food_name: str) -> Optional[Dict]:
    
    # Chuẩn hóa tên 
    food_name_normalized = food_name.lower().strip()
    
    # Tìm trong database
    nutrition = NUTRITION_DATABASE.get(food_name_normalized)
    
    if nutrition:
        # Thêm tên vào kết quả
        result = {'name': food_name, **nutrition}
        return result
    
    # Nếu không tìm thấy, trả về thông tin mặc định
    return {
        'name': food_name,
        'calories': None,
        'protein': None,
        'carbs': None,
        'fat': None,
        'fiber': None,
        'benefits': ['Thông tin dinh dưỡng chưa có sẵn'],
        'note': 'Đang cập nhật dữ liệu'
    }


def get_nutrition_with_fallback(food_name: str, api_key: str = None, app_id: str = None) -> Dict:
    
    return get_nutrition_info(food_name)

def format_nutrition_display(nutrition: Dict) -> str:
    #Format thông tin dinh dưỡng thành chuỗi đẹp
    html = f"<h4> {nutrition['name']}</h4>"
    
    if nutrition.get('calories'):
        html += f"<p><strong>Calories:</strong> {nutrition['calories']} kcal</p>"
    
    if nutrition.get('protein'):
        html += f"<p><strong>Protein:</strong> {nutrition['protein']}g</p>"
    
    if nutrition.get('carbs'):
        html += f"<p><strong>Carbs:</strong> {nutrition['carbs']}g</p>"
    
    if nutrition.get('fat'):
        html += f"<p><strong>Fat:</strong> {nutrition['fat']}g</p>"
    
    if nutrition.get('benefits'):
        html += "<p><strong>Lợi ích:</strong></p><ul>"
        for benefit in nutrition['benefits']:
            html += f"<li>{benefit}</li>"
        html += "</ul>"
    
    return html