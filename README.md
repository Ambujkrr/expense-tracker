# Expense Tracker

A full-stack personal Expense Tracker built with Python, Flask, MySQL, Chart.js and Scikit-learn.

## Features

- Add, edit and delete expenses
- Category and month filters
- Spending dashboard and charts
- Category-wise budgets and budget status
- Smart spending insights
- Next-month spending prediction using Linear Regression
- MAE-based model evaluation
- Expense anomaly detection using Isolation Forest

## Tech Stack

Python, Flask, MySQL, HTML/CSS, Chart.js, Scikit-learn

## Project Structure

```text
ExpenseTracker/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
└── templates/
    ├── index.html
    └── edit_expense.html
```

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Create a MySQL database named `expense_tracker` with the required `categories` and `expenses` tables.
4. Set the `MYSQL_PASSWORD` environment variable for your local MySQL password.
5. Run `python app.py`.
6. Open `http://127.0.0.1:5000`.

## Note

Database credentials are supplied through an environment variable and are not committed to the repository.
