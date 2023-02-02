import face_recognition
from sklearn import svm
import os
import cv2

#===================================== Train_Data =====================================#
# Đào tạo trình phân loại SVC
# Dữ liệu huấn luyện sẽ là tất cả các mã hóa khuôn mặt từ tất cả các hình ảnh đã biết và nhãn là tên của chúng
encodings = []
names = []

# Thư mục đào tạo
train_dir = os.listdir('train_dir/')

# Lặp qua từng người trong thư mục đào tạo
for person in train_dir:
    pix = os.listdir("train_dir/" + person)

    print(person + " đang được sử dụng để đào tạo")
    
    # Lặp qua từng hình ảnh đào tạo cho người hiện tại
    for person_img in pix:
        # Nhận mã hóa khuôn mặt cho khuôn mặt trong mỗi tệp hình ảnh
        face = face_recognition.load_image_file("train_dir/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # Nếu hình ảnh đào tạo chứa chính xác một khuôn mặt
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Thêm mã hóa khuôn mặt cho hình ảnh hiện tại với nhãn (tên) tương ứng vào dữ liệu đào tạo
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " đã bị bỏ qua vì không thể sử dụng để đào tạo")
# Tạo và đào tạo bộ phân loại SVC
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

#============================== Face Recognition ==============================#
face_names = []

# Load thư mục chứa ảnh dữ liệu Img_data
lt_path = os.listdir('Test')

# Nhận dạng khuôn mặt tìm được và ghi tên người trong ảnh ra file
f = open("Name_Found.txt", 'a', encoding = 'utf-8')

for i in range(0, len(lt_path)):
    # Tạo đường dẫn đến file ảnh
    img_path = "Test/" + lt_path[i]
    # Tải hình ảnh thử nghiệm có các khuôn mặt không xác định vào một mảng không rõ ràng
    test_image = face_recognition.load_image_file(img_path)
    # Tìm tất cả các khuôn mặt trong hình ảnh thử nghiệm bằng mô hình dựa trên HOG mặc định
    face_locations = face_recognition.face_locations(test_image)
    no = len(face_locations)
    f.write("===================# File: " + lt_path[i] + " #===================\n")
    f.write("Số khuôn mặt được phát hiện: " + str (no) + "\n")

    # Dự đoán tất cả các khuôn mặt trong hình ảnh thử nghiệm bằng bộ phân loại được đào tạo
    print("Tìm thấy:")

    for j in range(no):
        test_image_enc = face_recognition.face_encodings(test_image)[j]
        name = clf.predict([test_image_enc])
        print(*name)
        name_pp = str (*name)
        f.write("Tìm thấy: " + name_pp + "\n")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f.close()
# Giải phóng bộ nhớ cho các cửa sổ đã tạo ra
cv2.destroyAllWindows()



    

