
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class PredictionHistory(db.Model):
    """Lưu lịch sử phân tích"""
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Thông tin ảnh
    image_name = db.Column(db.String(255), nullable=False)
    image_data = db.Column(db.Text)  # Base64 string (optional)
    
    # Loại phân tích
    food_type = db.Column(db.String(50), nullable=False)  # 'ingredient' hoặc 'dish'
    
    # Model đã dùng
    model_name = db.Column(db.String(100), nullable=False)  # 'MobileNetV2' hoặc 'ResNet152V2'
    
    # Kết quả dự đoán (top 3)
    prediction_result = db.Column(db.Text, nullable=False)  # JSON string
    
    # Kết quả hàng đầu
    top_prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    
    # Metadata
    user_ip = db.Column(db.String(50))
    user_agent = db.Column(db.String(255))
    
    # Timestamp
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.top_prediction} ({self.confidence:.2f}%)>'
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'image_name': self.image_name,
            'food_type': self.food_type,
            'model_name': self.model_name,
            'predictions': json.loads(self.prediction_result),
            'top_prediction': self.top_prediction,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
        }

class Statistics(db.Model):
    """Thống kê tổng hợp"""
    __tablename__ = 'statistics'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Thống kê theo ngày
    date = db.Column(db.Date, default=datetime.utcnow().date, nullable=False, unique=True)
    
    # Số lượng dự đoán
    total_predictions = db.Column(db.Integer, default=0)
    ingredient_predictions = db.Column(db.Integer, default=0)
    dish_predictions = db.Column(db.Integer, default=0)
    
    # Top dự đoán (JSON)
    top_ingredients = db.Column(db.Text)  # JSON: {'apple': 15, 'banana': 12, ...}
    top_dishes = db.Column(db.Text)  # JSON: {'Pho': 20, 'Banh mi': 15, ...}
    
    # Timestamp
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Statistics {self.date}: {self.total_predictions} predictions>'
    
    def to_dict(self):
        return {
            'date': self.date.isoformat(),
            'total_predictions': self.total_predictions,
            'ingredient_predictions': self.ingredient_predictions,
            'dish_predictions': self.dish_predictions,
            'top_ingredients': json.loads(self.top_ingredients) if self.top_ingredients else {},
            'top_dishes': json.loads(self.top_dishes) if self.top_dishes else {},
        }

def init_db(app):
    """Initialize database"""
    db.init_app(app)
    
    with app.app_context():
        # Tạo tất cả bảng
        db.create_all()
        print(" Database initialized!")
        
        # Kiểm tra bảng
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()
        print(f"   Tables created: {table_names}")

def save_prediction(image_name, food_type, model_name, predictions, user_ip=None, user_agent=None):
    """
    Lưu kết quả prediction vào database
    
    Args:
        image_name: Tên file ảnh
        food_type: 'ingredient' hoặc 'dish'
        model_name: 'MobileNetV2' hoặc 'ResNet152V2'
        predictions: List of dicts [{'name': 'apple', 'confidence': 95.5}, ...]
        user_ip: IP address (optional)
        user_agent: User agent string (optional)
    """
    try:
        # Tạo record mới
        prediction = PredictionHistory(
            image_name=image_name,
            food_type=food_type,
            model_name=model_name,
            prediction_result=json.dumps(predictions),
            top_prediction=predictions[0]['name'],
            confidence=predictions[0]['confidence'],
            user_ip=user_ip,
            user_agent=user_agent,
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        # Cập nhật statistic
        update_statistics(food_type, predictions[0]['name'])
        
        return prediction.id
        
    except Exception as e:
        db.session.rollback()
        print(f" Error saving prediction: {e}")
        return None

def update_statistics(food_type, prediction_name):
    """Cập nhật thống kê hàng ngày"""
    try:
        today = datetime.utcnow().date()
        
        # Tìm hoặc tạo record cho ngày hôm nay
        stat = Statistics.query.filter_by(date=today).first()
        
        if not stat:
            stat = Statistics(
                date=today,
                total_predictions=0,
                ingredient_predictions=0,
                dish_predictions=0,
                top_ingredients='{}',
                top_dishes='{}',
            )
            db.session.add(stat)
        
        # Cập nhật counts
        stat.total_predictions += 1
        
        if food_type == 'ingredient':
            stat.ingredient_predictions += 1
            # Cập nhật top nguyên liệu
            top = json.loads(stat.top_ingredients) if stat.top_ingredients else {}
            top[prediction_name] = top.get(prediction_name, 0) + 1
            stat.top_ingredients = json.dumps(top)
        else:
            stat.dish_predictions += 1
            # Cập nhật top món ăn
            top = json.loads(stat.top_dishes) if stat.top_dishes else {}
            top[prediction_name] = top.get(prediction_name, 0) + 1
            stat.top_dishes = json.dumps(top)
        
        db.session.commit()
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating statistics: {e}")

def get_recent_predictions(limit=10):
    """Lấy predictions gần đây nhất"""
    return PredictionHistory.query.order_by(
        PredictionHistory.created_at.desc()
    ).limit(limit).all()

def get_statistics(days=7):
    """Lấy thống kê trong n ngày gần đây"""
    from datetime import timedelta
    start_date = datetime.utcnow().date() - timedelta(days=days-1)
    
    return Statistics.query.filter(
        Statistics.date >= start_date
    ).order_by(Statistics.date.asc()).all()

def get_overall_statistics():
    """Lấy thống kê tổng hợp"""
    total = PredictionHistory.query.count()
    ingredients = PredictionHistory.query.filter_by(food_type='ingredient').count()
    dishes = PredictionHistory.query.filter_by(food_type='dish').count()
    
    # Top 10 
    from sqlalchemy import func
    top_predictions = db.session.query(
        PredictionHistory.top_prediction,
        func.count(PredictionHistory.id).label('count')
    ).group_by(
        PredictionHistory.top_prediction
    ).order_by(
        func.count(PredictionHistory.id).desc()
    ).limit(10).all()
    
    return {
        'total_predictions': total,
        'ingredient_predictions': ingredients,
        'dish_predictions': dishes,
        'top_predictions': [
            {'name': name, 'count': count} 
            for name, count in top_predictions
        ]
    }