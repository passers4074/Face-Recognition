# Phát hiện và nhận dạng khuôn mặt với Dlib Deep Learning qua thư viện Face Recognition
Cài đặt:
 - Cài đặt Anaconda
 - Tạo môi trường (thư viện Face Recognition ổn định với phiên bản python 3.8 trên Anaconda): 
        conda create -n evn_face_two python=3.8
 - Chuyển sang môi trường mới:
        conda activate evn_face_two
 - Cài đặt thư viện Face Recognition: 
        conda install -c conda-forge face_recognition
 - Cài đặt thư viện OpenCV: 
        conda install -c conda-forge opencv

Các File, Folder và vai trò của chúng:

Thư mục:
 - Image: Chứa vài bức ảnh dùng để phát hiện khuôn mặt
 - Img_Data: Chứa các ảnh dùng để huấn luyện
 - train_dir: Chứa các thư mục con chứa các ảnh của từng người có tên là tên thư mục con. Dùng để huấn luyện
 - Test: Chứa 30 bức ảnh để nhận dạng
 - Video: Chứa 5 video dùng để nhận dạng
 
 File:
 - find_faces.py: Phát hiện và show các khuôn mặt phát hiện được trong thư mục Image
 - face_recognition_img.py: Huấn luyện và nhận dạng các khuôn mặt trong các bức ảnh ở thư mục Test và ghi thông tin ra file Name_Found.txt
 - face_recognition_video.py: Huấn luyện và nhận dạng các khuôn mặt trong video ở thư mục Video.
 - face_recognition_camera.py: Huấn luyện và nhận dạng các khuôn mặt trong video camera
