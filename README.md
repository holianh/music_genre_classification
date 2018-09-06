# music_genre_classification
https://challenge.zalo.ai/portal/music/final-leaderboard

# Make dir
$ chmod u+x make_directory.sh && ./make_directory.sh

# Download dataset
Tải file audio của tập train, public test và private test bỏ vào thư mục datasets/train datasets/test datasets/private_test

# Extract mel spectrogram features

Với tập train, mỗi file audio 2ph tách thành các đoạn 20s: với các class 1,9,10 do số lượng samples ít nên mình dịch từng đoạn 5s (có nghĩa 1 file 2ph tách thành 21 đoạn 20s), các class còn lại dịch từng đoạn 10s (1 file 2ph tách thành 11 file 20s). Mỗi đoạn 20s mình sẽ tính toán melspectrogram và lưu thành ảnh 512x512x3 

Tham khảo [https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html]

Với tập public test, tương tự chia file 2ph thành các đoạn 20s, độ dịch 5s, tính toán melspectrogram và lưu thành ảnh 512x512x3. Với tập private test, do có một số file có độ dài dưới 2ph nên mình sẽ concatenate cho đến khi chiều dài >= 2ph, sau đó cắt đúng độ dài 2ph, tương tự chia thành 11 đoạn 20s, dich 10s, lưu thảnh ảnh 512x512x3. Sử dụngtest time augmentation là horizontal flip, tổng cộng 1 file 2ph sẽ được tách thành 22 file nhỏ, predict cho 22 file này. Kết quả predict file 2ph sẽ là trung bình của kết quả predict 22 file nhỏ 20s.

Để tính melspectrogram và lưu thành fiel ành đơn giản chạy 2 file trong thư mục src

$ python audio2img.py 

$ python audio2img_private_testset.py

# Cross validation

Để generate ra 5 fole chạy file trong thư mục src

$ python create_trainset.py

Chia tập trainset thành 5 fold (chú ý là chia với tập audio 2ph). Ững với mỗi file audio sẽ sủ dụng 21 file ảnh (class 1,9,10) hoặc 11 file ảnh (class còn lại). Mình sử dụng 6 pretrain model trên tập image net: densenet169, densenet201, inception_resnet_v2, inception_v3, resnet50, xception. Warmup 1 epoch đầu, sử dụng augmentation horizontal flip.

Code train và predict cho mõi model nằm trong thư mục [model]/src, trong quá trình train weight sẽ nằm trong thư mục weights, training log sẽ nằm trong thư mục logs, data chứa output predict của mỗi model

# Bước 1: train lần 1

Train trên 2 fold 0,1, sử dụng 3 base model là densenet201, inception_v3, densenet169 (chọn 3 thằng này vì số lượng parameters ít nên train nhanh). Khi hội tụ val_acc của tập valid đã phân rã thành các ảnh nhỏ là tầm 0.6996 (^_^). Sau khi gom lại (predict mỗi ảnh nhỏ được probs 1x10, prob file 2ph = average ảnh nhỏ, chọn class có prob lớn nhất) thì val_acc tầm 0.79. Mình predict cho tập public test thì LB là 0.78.

# Bước 2: stacking lần 1

Predict cho cả 3 tập train, validation, public test trên các ảnh nhỏ, gom lại tương ứng với file lớn, thu được một 3 matrix có shape: train_sizex10, valid_sizex10, test_sizex10. Xếp chồng kết quả của 3 model theo chiều ngang được 3 matrix train_sizex3x10, valid_sizex3x10, test_sizex3x10. Tạo 1 model cnn đơn giản với input shape là 3x10, output shape là 1x10, các lớp cnn có kernel size là 2x1 [kiểu thay vì cộng trung bình thì đê cho thằng cnn tự tính tỉ lệ tốt nhất giữa 3 model]. Val_acc sau stack cho mỗi fold 0 và 1 khoảng 0.808. Cộng trung bình cho 2 fold 0 và 1, mình submit thì LB trên public test là 0.802

# Bước 3: pseudo labeling

Lúc này LB là 0.802 có nghĩa predict trên bộ public test có độ chính xác 80%. Chọn trên bộ public test các file có xác suất > 85%, đưa vào tập train train tiếp. Vi không có thời gian nên mình sử dụng 3 fold còn lại 2,3,4 để train với tập train + pseudo test set này. Tiến hành train giống bước 1, nhưng lần này train với cả 6 model, val_acc cho tập ảnh sau khi phân rã của mỗi model lúc này boost từ 0.6996 lên được 0.72.

# Bước 4: stacking lần 2

Predict cho cả 3 tập train + pseudo test, validation, public test trên các ảnh nhỏ, gom lại tương ứng với file lớn, thu được một 3 matrix có shape: (train_size_pseudo_size)x10, valid_sizex10, test_sizex10. Xếp chồng kết quả của 6 model theo chiều ngang được 3 matrix (train_size_pseudo_size)x6x10, valid_sizex6x10, test_sizex6x10.

- Tạo 2 model cnn với input shape là 6x10, output shape là 1x10 [1 thằng thì thêm một vài lớp skip connection cho khác với thằng còn lại], gọi 2 thằng này là cnn_stacking1 và cnn_stacking2

- Tạo 1 lgb model với input shape là 1x60, output shape là 1x10. Gọi bé này là lgb_stacking1

# Bước 5: đánh bừa trọng số :v

Đánh trọng số cho thằng cnn_staking1 là 0.45, cnn_stacking2 là 0.45 và bé lgb_stacking1 là 0.1

Kết quả cuối = 0.45 x cnn_stacking1 + 0.45 x cnn_stacking2 + 0.1 x lgb_stacking1.

Kết quả LB của tập publict test là 0.81593


Sau khi có tập private test thì final LB là 0.70144. Mình nghĩ distribution của bộ private và public test có thể hơi khác, việc xuất hiện một số file dưới 2ph mình cũng thấy hơi ngờ ngợ. Nhưng kệ cứ tin vào val_acc trên CV của mình và đợi kết quả cuối cùng, cũng may là vẫn TOP1 ^^.

Vì mình dành thời gian 3 tuần cho cuộc thi landmark nên chỉ dành đúng 3 ngày cho eda và train cho bài toán music, muốn thêm thì hết thời gian nên chịu. Nếu muốn improve accuracy thì mọi người có thể thử thêm mfcc và thêm base model.

Nếu thấy có ích thì cho mình một *. Tạm biệt và hẹn gặp lại ở cuộc thi sau (⊙_⊙)