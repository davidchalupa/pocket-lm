import json
import os
from datetime import datetime


class ECommerceOrderProcessor:
    """
    A classic 'God Class' that handles reading files, validating logic,
    doing math, and faking database connections.
    """

    def __init__(self, db_connection_string):
        self.db_string = db_connection_string
        self.tax_rate = 0.08

    def load_order_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def validate_order(self, order_data):
        if "items" not in order_data or not order_data["items"]:
            return False
        if "customer_id" not in order_data:
            return False
        return True

    def apply_discount(self, subtotal, discount_code):
        discounts = {
            "SAVE10": 0.10,
            "HALFOFF": 0.50
        }
        if discount_code in discounts:
            return subtotal - (subtotal * discounts[discount_code])
        return subtotal

    def calculate_total(self, order_data):
        subtotal = sum(item.get("price", 0) * item.get("qty", 1) for item in order_data["items"])

        discount_code = order_data.get("discount_code")
        if discount_code:
            subtotal = self.apply_discount(subtotal, discount_code)

        tax = subtotal * self.tax_rate
        return subtotal + tax

    def save_to_database(self, order_id, total, customer_id):
        # Simulating a DB write
        timestamp = datetime.now().isoformat()
        db_record = f"INSERT INTO orders (id, customer, total, date) VALUES ('{order_id}', '{customer_id}', {total}, '{timestamp}')"
        print(f"[DB LOG] Connecting to {self.db_string}...")
        print(f"[DB LOG] Executing: {db_record}")
        return True

    def process_order(self, filepath):
        print(f"Starting processing for {filepath}")
        data = self.load_order_data(filepath)

        if not self.validate_order(data):
            print("Invalid order data.")
            return False

        final_total = self.calculate_total(data)

        order_id = data.get("order_id", "UNKNOWN")
        customer_id = data.get("customer_id")

        success = self.save_to_database(order_id, final_total, customer_id)
        if success:
            print(f"Order {order_id} processed successfully. Total: ${final_total:.2f}")
        return success


if __name__ == "__main__":
    # Fake usage
    processor = ECommerceOrderProcessor("postgres://user:pass@localhost:5432/ecommerce")
    # processor.process_order("sample_order.json")
