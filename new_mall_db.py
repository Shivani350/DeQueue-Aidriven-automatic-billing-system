import sqlite3

def deploy_database():
    conn = sqlite3.connect('mall.db')
    cursor = conn.cursor()
    
    # Create the Three-Table Architecture
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS products (
            barcode TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            price REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cart (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            barcode TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (barcode) REFERENCES products(barcode)
        );
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_amount REAL,
            items_count INTEGER,
            sale_date DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # Deployment Inventory
    products = [
        ('101', 'Amul Milk 1L', 66.00),
        ('102', 'Brown Bread 400g', 50.00),
        ('103', 'Eggs 12pcs', 90.00),
        ('104', 'Cadbury Dairy Milk', 40.00),
        ('105', 'Mineral Water 1L', 20.00)
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?)", products)
    conn.commit()
    conn.close()
    print("Database mall.db created successfully!")

if __name__ == "__main__":
    deploy_database()