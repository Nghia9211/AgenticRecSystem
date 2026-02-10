import json
import os

def combine_user_groundtruth(folder_path, output_file):
    combined_data = []

    # Lặp qua các index từ 0 đến 99
    for i in range(100):
        task_filename = f"tasks/task_{i}.json"
        gt_filename = f"groundtruth/groundtruth_{i}.json"
        
        # Đường dẫn đầy đủ tới file
        task_path = os.path.join(folder_path, task_filename)
        gt_path = os.path.join(folder_path, gt_filename)

        try:
            # Đọc file task để lấy user_id
            with open(task_path, 'r', encoding='utf-8') as f_task:
                task_data = json.load(f_task)
                user_id = task_data.get("user_id")

            # Đọc file groundtruth để lấy item_id
            with open(gt_path, 'r', encoding='utf-8') as f_gt:
                gt_data = json.load(f_gt)
                item_id = gt_data.get("ground truth")

            # Thêm vào danh sách kết quả
            if user_id and item_id:
                combined_data.append({
                    "user_id": user_id,
                    "item_id": item_id
                })
        
        except FileNotFoundError:
            print(f"Cảnh báo: Không tìm thấy file {task_filename} hoặc {gt_filename}")
        except Exception as e:
            print(f"Lỗi khi xử lý file index {i}: {e}")

    # Ghi kết quả ra file JSON mới
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(combined_data, f_out, indent=4, ensure_ascii=False)
    
    print(f"Hoàn thành! Đã lưu {len(combined_data)} bản ghi vào file {output_file}")

# Sử dụng hàm (Thay '.' bằng đường dẫn thư mục chứa các file của bạn)
combine_user_groundtruth(folder_path=r'C:\Users\Admin\Desktop\Document\AgenticCode\AgentRecBench\dataset\task\user_cold_start\yelp', output_file='final_mask_yelp.json')