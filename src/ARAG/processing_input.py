import json
from datetime import datetime
import sys

class ReviewProcessor:
    DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
    SECONDS_IN_A_DAY = 24 * 60 * 60

    def __init__(self):
        self.all_reviews = []
        self.sorted_reviews = []
        self.short_term_context = []
        self.long_term_context = []

    @staticmethod
    def _get_sort_key(review):
        date_str = review.get("date_updated")
        timestamp_int = 0

        if not date_str:
            print(f"No date field 'date_updated'. ID: {review.get('review_id', 'N/A')}")
        else:
            try:
                dt_object = datetime.strptime(date_str, ReviewProcessor.DATE_FORMAT)
                timestamp_int = int(dt_object.timestamp())
            except ValueError as e:
                print(f"Error '{date_str}' (ID: {review.get('review_id', 'N/A')}): {e}.")
        
        review['date_updated_int'] = timestamp_int
        return timestamp_int

    def load_reviews(self, input_file):
        self.all_reviews = input_file
        return True

    def process_and_split(self, days_window_i, max_items_k, max_items_m):
        if not self.all_reviews:
            print("Please load Reviews first!")
            return

        self.sorted_reviews = sorted(self.all_reviews, key=self._get_sort_key, reverse=True)
        
        if not self.sorted_reviews:
            return
        
        session_duration_seconds = days_window_i * self.SECONDS_IN_A_DAY
        latest_timestamp = self.sorted_reviews[0].get('date_updated_int', 0)
        time_threshold = latest_timestamp - session_duration_seconds

        time_window_reviews = []
        long_term_reviews_temp = []

        for review in self.sorted_reviews:
            review_timestamp = review.get('date_updated_int', 0)
            if review_timestamp > 0 and review_timestamp >= time_threshold:
                time_window_reviews.append(review)
            else:
                long_term_reviews_temp.append(review)
        

        self.short_term_context = time_window_reviews[:max_items_k]
        self.long_term_context = long_term_reviews_temp[:max_items_m]

        print("\n--- Splited Information ---")
        print(f"Most Recent Timestamp : {datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Time stamp (i={days_window_i} days before): {datetime.fromtimestamp(time_threshold).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Limit K (short-term): {max_items_k} items.")
        print(f"Limit M (long-term): {max_items_m} items.")
        print(f" 'Short Term Context': {len(self.short_term_context)} items.")
        print(f" 'Long Term Context': {len(self.long_term_context)} item.")

    @staticmethod
    def save_to_json(data, filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"SUCCESS : Write {len(data)} items into : {filename}")
        except Exception as e:
            print(f"ERROR while write JSON '{filename}': {e}")
