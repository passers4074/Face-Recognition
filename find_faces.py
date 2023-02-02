from PIL import Image
import face_recognition
import cv2
import os

# Load thư mục chứa ảnh dữ liệu Img_data
lt_path = os.listdir('Image')

for i in range(0, len(lt_path)):
    # Tạo đường dẫn đến file ảnh
    img_path = "Image/" + lt_path[i]
    img = cv2.imread(img_path)
    # Tìm tất cả các khuôn mặt trong hình ảnh bằng cách sử dụng mạng nơ-ron phức hợp được đào tạo trước.
    face_locations = face_recognition.face_locations(img, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    for face_location in face_locations:
        # In vị trí của từng khuôn mặt trong hình ảnh này
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        
        face_image = img[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
    
# Giải phóng bộ nhớ cho các cửa sổ đã tạo ra
cv2.destroyAllWindows()