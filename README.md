# Expense Tracker

A full-stack personal expense tracker built with Python, Flask, MySQL, Chart.js and
scikit-learn. Every registered user gets their own categories, budgets and expenses;
accounts are fully isolated from each other.

## Features

- Registration, login and logout (username or email plus a password)
- Multi-user expense ownership: one account can never view or edit another account's data
- Add, edit and delete expenses with input validation (positive amounts, real dates,
  notes up to 255 characters)
- Six default categories created automatically for each new account
- Per-category monthly budgets with an inline set/update form and status bars
- Dashboard analytics: totals, expense counts, category-wise and monthly summaries
- Interactive charts (category doughnut and monthly trend line) with Chart.js
- Next-month spending prediction using Linear Regression (needs 2+ months of data)
- MAE-based model evaluation holding out the latest month (needs 4+ months of data)
- Expense anomaly detection using Isolation Forest (needs 5+ expenses)
- Smart spending insights (top category, budget warnings, month-over-month trend)
- CSRF protection on every form, application-wide
- Global error handling with user-friendly 404 and 500 pages; stack traces are logged
  server-side and never shown to the user
- Health endpoint for deployment checks
- Automated test suite that runs against an in-memory fake database (no MySQL needed)

## Tech Stack

Python, Flask, MySQL, HTML/CSS, Chart.js, scikit-learn

## Project Structure

```text
ExpenseTracker/
├── app.py                      # Flask app: routes, ML, error handling
├── requirements.txt
├── .gitignore
├── README.md
├── conftest.py                 # Shared pytest fixtures (fake database)
├── templates/
│   ├── index.html              # Dashboard, charts, budgets, insights, anomalies
│   ├── login.html
│   ├── register.html
│   └── edit_expense.html
└── tests/
    ├── test_validation.py      # Expense add/update input validation
    ├── test_ownership.py       # Cross-user access is blocked
    ├── test_budget.py          # Category budget setting
    └── test_error_handlers.py  # Global 404/500 handlers
```

## Database

Create a MySQL database and apply this schema once:

```sql
CREATE TABLE users (
    id            INT NOT NULL AUTO_INCREMENT,
    username      VARCHAR(50)  NOT NULL,
    email         VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_users_username (username),
    UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB;

CREATE TABLE categories (
    id             INT NOT NULL AUTO_INCREMENT,
    user_id        INT NOT NULL,
    name           VARCHAR(50)   NOT NULL,
    monthly_budget DECIMAL(10,2) NOT NULL DEFAULT 0.00,
    PRIMARY KEY (id),
    UNIQUE KEY uq_categories_user_name (user_id, name),
    CONSTRAINT fk_categories_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE expenses (
    id           INT NOT NULL AUTO_INCREMENT,
    user_id      INT NOT NULL,
    category_id  INT NOT NULL,
    amount       DECIMAL(10,2) NOT NULL,
    note         VARCHAR(255) NOT NULL,
    expense_date DATE NOT NULL,
    PRIMARY KEY (id),
    KEY idx_expenses_user_category (user_id, category_id),
    KEY idx_expenses_user_date (user_id, expense_date),
    CONSTRAINT fk_expenses_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB;
```

Connection details come from environment variables only — no credentials are stored
in the repository:

| Variable | Required | Description |
| --- | --- | --- |
| `MYSQL_HOST` | No | MySQL host (defaults to localhost) |
| `MYSQL_PORT` | No | MySQL port (defaults to `3306`) |
| `MYSQL_USER` | No | MySQL user (avoid using `root`) |
| `MYSQL_PASSWORD` | Yes | MySQL password |
| `MYSQL_DATABASE` | No | Database name (defaults to `defaultdb`) |
| `SECRET_KEY` | Production | Flask session key; a random one is generated in development |
| `FLASK_ENV` | No | Set to `production` to enable production settings |
| `FLASK_DEBUG` | No | Set to `1` to enable debug mode |

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Create a MySQL database and apply the schema above.
4. Export the `MYSQL_*` environment variables for your connection.
5. Run `python app.py`.
6. Open `http://127.0.0.1:5000` and register an account.

## Tests

The suite uses an in-memory fake database, so no MySQL server is required:

```bash
python -m pytest tests -p no:cacheprovider
```

## Deployment

For production, set `FLASK_ENV=production` and a strong `SECRET_KEY`, then serve with
gunicorn:

```bash
gunicorn app:app
```

## Notes

- Database credentials are supplied through environment variables and are never committed.
- Passwords are stored as Werkzeug password hashes (scrypt/PBKDF2), never as plaintext.
- The `-p no:cacheprovider` flag keeps pytest from writing a cache directory.
