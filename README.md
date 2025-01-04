# AI Database Query Assistant

This tool converts natural language to SQL queries, visualizes data, and generates reports.

## Setup Instructions

1. **Install Dependencies:**

   Create a virtual environment and install the dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```


2. **Create Demo Database:**

   ```bash
   python create_database.py
   ```
   The database created use a seed to make sure the data is consistent.

3. **Start Server:**
   ```bash
   uvicorn main:app --reload
   ```

4. **Run Tests:**
   The command below will run the tests and generate a report in the tests/logs directory.
   If everything is working, you should see a PASS for each test.
   ```bash
   python -m pytest tests/test_llm_queries.py
   ```

## Example Queries

1. **Using curl:**

   ```bash
   curl -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
              "query": "What are the top 5 selling products by total revenue?",
              "db_connection": "sqlite:///demo.db"
            }'
   ```

2. **Using Python:**

   ```python
   import requests

   response = requests.post(
       "http://localhost:8000/query",
       json={
           "query": "What are the top 5 selling products by total revenue?",
           "db_connection": "sqlite:///demo.db"
       }
   )
   print(response.json())
   ```

## Sample Queries

- Show me the top 5 products by revenue
- What is the total sales for each product category?
- Which products have sales over $10,000?
- Show monthly sales trends for 2024
- Compare sales between different product categories

## Database Connections

| Database   | Connection String                                  |
| ---------- | -------------------------------------------------- |
| SQLite     | `sqlite:///database.db`                            |
| PostgreSQL | `postgresql://user:password@localhost:5432/dbname` |
| MySQL      | `mysql://user:password@localhost:3306/dbname`      |

## Project Structure

| File                 | Description               |
| -------------------- | ------------------------- |
| `main.py`            | FastAPI application       |
| `create_database.py` | Demo database generator   |
| `reports/`           | Generated HTML reports    |
| `.env`               | Environment configuration |
| `info.txt`           | This information file     |

## Troubleshooting

### 1. API Key Issues

- Check `.env` file exists
- Verify API key is correct
- Try: `export OPENAI_API_KEY='your-key'`

### 2. Database Issues

- Verify database exists
- Check connection string
- Check file permissions

### 3. Report Issues

- Check `reports/` directory exists
- Verify plotly installation
- Check browser compatibility

---

For more details, visit: [project repository URL]
