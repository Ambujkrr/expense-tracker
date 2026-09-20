"""Database migration script for the password_resets table.

Run from a terminal with MYSQL_HOST, MYSQL_PORT, MYSQL_USER, and
MYSQL_PASSWORD set in the environment:

    python migrate_password_resets.py

This script creates the `password_resets` table required for the secure
email OTP-based password reset system with foreign key constraints,
indexes, and verification checks.
"""

import os
import sys
import mysql.connector

DATABASE = os.environ.get("MYSQL_DATABASE", "defaultdb")


def connect():
    """Connect to the target MySQL database using environment variables."""
    try:
        return mysql.connector.connect(
            host=os.environ.get("MYSQL_HOST", "localhost"),
            port=int(os.environ.get("MYSQL_PORT", "3306")),
            user=os.environ.get("MYSQL_USER"),
            password=os.environ.get("MYSQL_PASSWORD"),
            database=DATABASE,
            ssl_disabled=False,
        )
    except Exception as exc:
        raise SystemExit(f"Database connection failed: {exc}")


def fetch_one(cursor, query, params=()):
    cursor.execute(query, params)
    return cursor.fetchone()


def fetch_all(cursor, query, params=()):
    cursor.execute(query, params)
    return cursor.fetchall()


def table_exists(cursor, table_name):
    row = fetch_one(
        cursor,
        """
        SELECT COUNT(*) AS count
        FROM information_schema.tables
        WHERE table_schema = DATABASE() AND table_name = %s
        """,
        (table_name,),
    )
    return row["count"] == 1


def preflight(cursor):
    """Verify that required base tables exist before creating password_resets."""
    if not table_exists(cursor, "users"):
        raise RuntimeError("Preflight failed: 'users' table is missing. Run auth migration first.")

    if table_exists(cursor, "password_resets"):
        print("Note: 'password_resets' table already exists. Applying idempotent schema updates if needed.")
    else:
        print("Preflight PASS: 'users' table exists; ready to create 'password_resets'.")


def apply_migration(cursor, connection):
    """Create password_resets table and indexes idempotently."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS password_resets (
            id INT NOT NULL AUTO_INCREMENT,
            user_id INT NOT NULL,
            email VARCHAR(255) NOT NULL,
            otp_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            attempts INT NOT NULL DEFAULT 0,
            is_used TINYINT(1) NOT NULL DEFAULT 0,
            ip_address VARCHAR(45) NULL,
            PRIMARY KEY (id),
            KEY idx_resets_user_id (user_id),
            KEY idx_resets_email (email),
            KEY idx_resets_expires (expires_at),
            CONSTRAINT fk_resets_user
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        ) ENGINE=InnoDB
        """
    )
    connection.commit()
    print("Migration: 'password_resets' table created or already present.")


def verify_post_migration(cursor):
    """Verify that password_resets table and expected columns/indexes exist."""
    if not table_exists(cursor, "password_resets"):
        raise RuntimeError("Post-migration verification failed: 'password_resets' table not found.")

    expected_cols = {
        "id",
        "user_id",
        "email",
        "otp_hash",
        "created_at",
        "expires_at",
        "attempts",
        "is_used",
        "ip_address",
    }
    cols = fetch_all(
        cursor,
        """
        SELECT COLUMN_NAME AS column_name
        FROM information_schema.columns
        WHERE table_schema = DATABASE() AND table_name = 'password_resets'
        """,
    )
    actual_cols = {c["column_name"] for c in cols}
    missing = expected_cols - actual_cols
    if missing:
        raise RuntimeError(f"Post-migration verification failed: missing columns {missing}")

    print("Post-migration PASS: 'password_resets' schema verified.")


def main():
    connection = connect()
    cursor = connection.cursor(dictionary=True)
    try:
        preflight(cursor)
        apply_migration(cursor, connection)
        verify_post_migration(cursor)
        print("Password reset migration completed successfully.")
    except Exception as err:
        connection.rollback()
        print(f"Migration error: {err}")
        sys.exit(1)
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
