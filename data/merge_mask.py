import json
import os

def merge_final_json_files(input_files, output_filename):
    merged_list = []

    for file_name in input_files:
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Kiểm tra nếu data là list thì nối vào, nếu là dict thì append
                    if isinstance(data, list):
                        merged_list.extend(data)
                    else:
                        merged_list.append(data)
                        
                print(f"✅ Đã đọc xong file: {file_name}")
            except Exception as e:
                print(f"❌ Lỗi khi đọc file {file_name}: {e}")
        else:
            print(f"⚠️ Không tìm thấy file: {file_name}")

    # Ghi toàn bộ dữ liệu đã gộp vào file cuối cùng
    with open(output_filename, 'w', encoding='utf-8') as f_out:
        json.dump(merged_list, f_out, indent=4, ensure_ascii=False)

    print(f"\n--- THÀNH CÔNG ---")
    print(f"Tổng số bản ghi sau khi gộp: {len(merged_list)}")
    print(f"Kết quả lưu tại: {output_filename}")

# --- CẤU HÌNH TÊN FILE CỦA BẠN Ở ĐÂY ---
files_to_merge = [
    "graph_data/final_mask_amazon.json", 
    "graph_data/final_mask_goodreads.json", 
    "graph_data/final_mask_yelp.json"
]

merge_final_json_files(files_to_merge, "ground_truth.json")