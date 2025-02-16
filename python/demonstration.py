import duckdb

# Create users and orders relations
users = duckdb.sql("SELECT 1 AS id, 'Alice' AS name, 25 AS age")
orders = duckdb.sql("SELECT 1 AS order_id, 1 AS user_id, 100 AS amount")

# Function where we explicitly register the variables we want to use
def join_users_orders_restricted(users_rel, orders_rel):
    # Register relations as temporary views
    duckdb.sql("CREATE TEMP VIEW my_users AS SELECT * FROM users_rel")
    duckdb.sql("CREATE TEMP VIEW my_orders AS SELECT * FROM orders_rel")

    print("\n🔹 Trying to join using my_users and my_orders (expected to work)...")
    result = duckdb.sql("""
        SELECT my_users.id, my_users.name, my_orders.amount 
        FROM my_users 
        LEFT JOIN my_orders ON my_users.id = my_orders.user_id
    """)
    print(result.df())

    print("\n🔹 Trying to join using the original variable names (expected to fail)...")
    try:
        result = duckdb.sql("""
            SELECT users.id, users.name, orders.amount 
            FROM users 
            LEFT JOIN orders ON users.id = orders.user_id
        """)
        print("❌ Unexpected Success:", result.df())
    except Exception as e:
        print("✅ Failed as expected:", e)

# Call function with restricted scope
join_users_orders_restricted(users, orders)
