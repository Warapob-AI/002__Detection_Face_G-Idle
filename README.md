# การจดจำใบหน้าสมาชิกวง (G)I-DLE
[![image.png](https://i.postimg.cc/4xFs7ZRy/image.png)](https://postimg.cc/xXHWhwCV)
โปรเจคนี้ใช้ Machine Learning และ Deep Learning ในการตรวจจับและจดจำใบหน้าของสมาชิกวง (G)I-DLE โดยใช้ SVM (Support Vector Machine) และ DeepFace เพื่อดึงคุณลักษณะจากใบหน้า

## คุณสมบัติหลัก
- ตรวจจับใบหน้าจากภาพโดยใช้โมเดล Caffe SSD
- ดึงคุณลักษณะใบหน้าด้วย DeepFace (Facenet)
- ฝึกโมเดล SVM เพื่อจำแนกใบหน้าเป็นสมาชิกแต่ละคน
- ทดสอบโมเดลกับภาพใหม่และแสดงผลการทำนาย

## การติดตั้ง
### ติดตั้งไลบรารีที่จำเป็น
ก่อนใช้งาน ให้ติดตั้งไลบรารีที่ต้องใช้โดยรันคำสั่งนี้:
```bash
pip install -r requirements.txt
```

## วิธีการใช้งาน
### 1. รันไฟล์ main.py
สคริปต์จะ:
- โหลดและตรวจจับใบหน้าจากภาพในแต่ละโฟลเดอร์
- แปลงใบหน้าเป็นเวกเตอร์คุณลักษณะ
- ฝึกโมเดล SVM และบันทึกโมเดล

### 2. ทดสอบโมเดลกับภาพใหม่
```bash
predict_new_image('test_image.png')

# ใส่ภาพลงไปใน predict_new_image()
```
โมเดลจะ:
- ตรวจจับใบหน้าในภาพที่ระบุ
- ทำนายว่าเป็นสมาชิกคนใดจากวง (G)I-DLE
- แสดงผลลัพธ์การทำนาย

## หมายเหตุ
- แนะนำให้ใช้รูปภาพที่มีใบหน้าชัดเจน ไม่มีสิ่งกีดขวาง เช่น แว่นตาหรือหมวก

📌 **โปรเจคนี้พัฒนาขึ้นเพื่อการศึกษาด้าน AI และ Machine Learning เท่านั้น**

