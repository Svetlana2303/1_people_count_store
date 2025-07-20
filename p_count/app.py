from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import uuid
import logging

app = Flask(__name__)
model = YOLO('yolov8n.pt')

logging.basicConfig(filename='logs.log', level=logging.INFO)

os.makedirs('static', exist_ok=True)
conn = sqlite3.connect('visitors.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS counts
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   date TEXT,
                   count INTEGER,
                   image_path TEXT)''')
conn.commit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    try:
        logging.info(f"Начало обработки в {datetime.now()}")

        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)
        
        results = model(img)
        count = sum(1 for box in results[0].boxes if box.cls == 0)  # Только люди

        result_filename = f'result_{uuid.uuid4().hex}.jpg'
        result_path = f'static/{result_filename}'
        output_img = results[0].plot()  # Рисуем bounding boxes
        cv2.imwrite(result_path, output_img)

        cursor.execute("INSERT INTO counts (date, count, image_path) VALUES (?, ?, ?)",
                      (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, result_path))
        conn.commit()
        
        return jsonify({
            'count': count,
            'image_url': result_path
        })

    except Exception as e:
        logging.error(f"Ошибка: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def history():
    cursor.execute("SELECT date, count, image_path FROM counts ORDER BY date DESC")
    data = cursor.fetchall()
    return render_template('history.html', history=data)

if __name__ == '__main__':
    app.run(debug=True)