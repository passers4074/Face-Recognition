import face_recognition
import cv2
import os
import numpy as np

#===================== TrainData =====================#
known_face_names = []
known_face_encodings  = []

# Load thư mục chứa ảnh dữ liệu
lt_path = os.listdir('Img_Data')

for i in range(0, len(lt_path)):
    # Tạo đường dẫn đến file ảnh
    img_path = "Img_Data/" + lt_path[i]
    # Đặt tên cho khuôn mặt
    name = lt_path[i]
    # Tải, học cách nhận dạng; thêm tên vào mảng mã hóa
    name_img = face_recognition.load_image_file(img_path)
    known_face_names.append(name)
    try:
        name_face_encoding = face_recognition.face_encodings (name_img)[0]
    except IndexError:
        height, width, _ = name_img.shape
        face_location = (0, width, height, 0)
        name_face_encoding = face_recognition.face_encodings(name_img, known_face_locations=[face_location])
    known_face_encodings.append(name_face_encoding)

#print("known_face_names: ", known_face_names)
#print("known_face_encodings: ", known_face_encodings)
print("Hoàn thành cập nhật dữ liệu. Bắt đầu nhận dạng...")

#===================== Face Recognition =====================#
# Đọc video từ camera
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Lấy một freame video
    ret, frame = video_capture.read()

    # Thay đổi kích thước khung hình của video thành kích thước 1/4 để xử lý nhận dạng khuôn mặt nhanh hơn
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Chuyển đổi hình ảnh từ màu BGR (OpenCV sử dụng) sang màu RGB (face_recognition sử dụng)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Chỉ xử lý mọi khung hình khác của video để tiết kiệm thời gian
    if process_this_frame:
        # Tìm tất cả các khuôn mặt và mã hóa khuôn mặt trong khung hình hiện tại của video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Xem khuôn mặt có khớp với (các) khuôn mặt đã biết không
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Nếu tìm thấy kết quả trùng khớp trong known_face_encodings, chỉ cần sử dụng kết quả đầu tiên.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Hoặc thay vào đó, sử dụng khuôn mặt đã biết với khoảng cách nhỏ nhất đến khuôn mặt mới
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Hiển thị kết quả
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Chỉnh tỷ lệ vị trí khuôn mặt vì khung mà chúng tôi phát hiện trong đã được thu nhỏ thành 1/4 kích thước
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Vẽ một hộp xung quanh khuôn mặt
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Vẽ nhãn có tên bên dưới khuôn mặt
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Hiển thị hình ảnh kết quả
    cv2.imshow('Video', frame)

    # Ghi tên người trong ảnh ra file write.txt
    f = open("write.txt",'w')
    f.write(name)
    f.close()
    
    # Nhấn 'q' trên bàn phím để thoát!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()