import json
from datetime import datetime
import os 

# --- 1. Hàm xử lý thời gian (Giữ nguyên) ---
class TimestampProcessor:
    @staticmethod
    def get_normalized_timestamp(data, source):
        # ... (Nội dung hàm giữ nguyên)
        try:
            if source == 'amazon':
                return int(data.get('timestamp', 0))
                
            elif source == 'yelp':
                date_str = data.get('date')
                if date_str:
                    dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                    return int(dt.timestamp() * 1000)
                    
            elif source == 'goodreads':
                date_str = data.get('date_added') or data.get('date_updated')
                if date_str:
                    try:
                        # Thử với múi giờ
                        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y") 
                        return int(dt.timestamp() * 1000)
                    except ValueError:
                        # Thử lại nếu không có múi giờ hoặc định dạng khác (nếu cần)
                        pass 
            return 0
        except Exception as e:
            return 0

# --- 2. Hàm xử lý chính (ĐÃ SỬA PHẦN ĐỌC REVIEW DATA) ---

def prepare_sasrec_data(ground_truth_file, review_data_file, output_file="sasrec_train_data.json"):
    
    # 1. Tải và chuẩn bị Ground Truth (Giữ nguyên)
    print(f"1. Tải Ground Truth từ {ground_truth_file}...")
    try:
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth_list = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {ground_truth_file}. Vui lòng kiểm tra đường dẫn.")
        return
    except json.JSONDecodeError:
        # Ground Truth thường là mảng, nên nếu lỗi JSON là do định dạng sai.
        print(f"Lỗi: File {ground_truth_file} không phải là JSON hợp lệ.")
        return
        
    gt_map = {item["user_id"]: item["item_id"] for item in ground_truth_list}
    print(f"   -> Đã tải {len(gt_map)} ground truth users.")

    # 2. Tải và chuẩn hóa Review Data (ĐÃ SỬA)
    print(f"2. Tải Review Data từ {review_data_file} (JSON Lines)...")
    
    normalized_reviews = []
    
    try:
        # *** PHẦN ĐỌC FILE ĐÃ ĐƯỢC SỬA ĐỔI ĐỂ XỬ LÝ JSON LINES ***
        with open(review_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Bỏ qua dòng trống
                if not line.strip():
                    continue
                    
                try:
                    # Dùng json.loads() để parse TỪNG DÒNG JSON
                    data = json.loads(line)
                    
                    # Dựa trên mẫu của bạn, key "source" nằm ngay trong đối tượng review
                    source = data.get('source')
                    if not source:
                         print(f"Cảnh báo: Review thiếu trường 'source'. Bỏ qua dòng: {line.strip()[:50]}...")
                         continue
                         
                    # Thêm source và tính timestamp
                    data['normalized_timestamp'] = TimestampProcessor.get_normalized_timestamp(data, source)
                    
                    normalized_reviews.append(data)
                    
                except json.JSONDecodeError as e:
                    print(f"Cảnh báo: Lỗi JSONDecodeError khi đọc dòng: {line.strip()[:50]}... Chi tiết: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {review_data_file}. Vui lòng kiểm tra đường dẫn.")
        return

    print(f"   -> Đã tải và chuẩn hóa {len(normalized_reviews)} reviews.")
    
    # 3. Tìm timestamp của Ground Truth Item cho mỗi User (Giữ nguyên logic)
    # ... (Phần còn lại của hàm giữ nguyên) ...
    gt_timestamps = {}
    
    print("3. Tìm timestamp của Ground Truth Item...")
    
    for review in normalized_reviews:
        user_id = review.get('user_id')
        item_id = review.get('item_id')
        
        if user_id in gt_map and item_id == gt_map[user_id]:
            current_ts = review['normalized_timestamp']
            if user_id not in gt_timestamps or current_ts > gt_timestamps[user_id]:
                 gt_timestamps[user_id] = current_ts

    print(f"   -> Đã tìm thấy GT timestamp cho {len(gt_timestamps)} user (trong tổng số {len(gt_map)} GT user).")
    
    # 4. Lọc Review Data (Giữ nguyên logic)
    
    print("4. Lọc Review Data...")
    
    filtered_reviews = []
    skipped_count = 0
    
    for review in normalized_reviews:
        user_id = review.get('user_id')
        timestamp = review['normalized_timestamp']
        
        if user_id in gt_timestamps:
            gt_timestamp = gt_timestamps[user_id]
            
            if timestamp <= gt_timestamp:
                filtered_reviews.append({
                    "user_id": review['user_id'],
                    "item_id": review['item_id'],
                    "timestamp": review['normalized_timestamp'],
                    "source": review['source'] 
                })
            else:
                skipped_count += 1
                
    print(f"   -> Đã giữ lại {len(filtered_reviews)} reviews.")
    print(f"   -> Đã loại bỏ {skipped_count} reviews (timestamp > GT timestamp).")
    
    # 5. Lưu kết quả (Giữ nguyên logic)
    
    print(f"5. Lưu kết quả vào {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_reviews, f, indent=4)
        
    print("✅ Hoàn tất quá trình chuẩn bị dữ liệu.")

# --- 3. Script chạy chính thức (Giữ nguyên) ---

if __name__ == '__main__':
    # THAY THẾ HAI BIẾN NÀY BẰNG ĐƯỜNG DẪN FILE THẬT CỦA BẠN
    # Đảm bảo bạn đang sử dụng tên file chính xác (ví dụ: review.json)
    GROUND_TRUTH_FILE_PATH = 'ground_truth.json' 
    REVIEW_DATA_FILE_PATH = 'review.json' 
    OUTPUT_FILE_PATH = 'sasrec_train_data.json'
    
    prepare_sasrec_data(
        ground_truth_file=GROUND_TRUTH_FILE_PATH, 
        review_data_file=REVIEW_DATA_FILE_PATH,
        output_file=OUTPUT_FILE_PATH
    )