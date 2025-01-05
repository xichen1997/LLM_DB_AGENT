import os
import json
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
import sys
import os
import numpy as np

# Add the parent directory to the Python path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import AIDBAgent

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///../sales_database.db')

# Test configuration
TEST_CONFIG = {
    'model': 'ollama',
    'temperature': 0.0,  # Set temperature to 0 for deterministic outputs
    'database_url': DATABASE_URL
}

# Create necessary directories
os.makedirs('tests', exist_ok=True)
os.makedirs('tests/logs', exist_ok=True)

@pytest.fixture
def ai_agent():
    """Create an AIDBAgent instance for testing with controlled parameters"""
    agent = AIDBAgent(model=TEST_CONFIG['model'])
    # Set Ollama parameters for deterministic output
    agent.ollama_params = {
        'temperature': TEST_CONFIG['temperature'],
        'num_predict': 256,  # Reasonable limit for SQL queries
        'num_ctx': 4096,     # Extended context window
        'stop': [';', '\n\n']  # Stop at query end or double newline
    }
    agent.connect_to_db(TEST_CONFIG['database_url'])
    return agent

def test_product_sales_pie_chart(ai_agent):
    """Test generating a pie chart for product sales"""
    # Load standard data
    with open('tests/product_and_sales_standard.json', 'r') as f:
        standard = json.load(f)
    
    query = "A pie chart of products and their subtotal sales."
    
    # Generate and execute SQL query
    sql_query = ai_agent.generate_sql_query(query)
    print("\nGenerated SQL Query:")
    print(sql_query)
    
    df = ai_agent.execute_query(sql_query)
    
    # Ensure DataFrame has the correct column names
    if 'total_amount' in df.columns:
        df = df.rename(columns={'total_amount': 'subtotal_sales'})
    
    print("\nDataFrame Info:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())
    
    # Generate visualization
    viz = ai_agent.generate_visualization(df, query)
    
    # Convert DataFrame to same format as standard for comparison
    df_dict = df.to_dict('records')
    
    print("\nTest Data:")
    for item in sorted(df_dict, key=lambda x: x['name']):
        print(f"{item['name']}: {item['subtotal_sales']}")
    
    print("\nStandard Data:")
    for item in sorted(standard['data'], key=lambda x: x['name']):
        print(f"{item['name']}: {item['subtotal_sales']}")
    
    # Sort both lists by product name for comparison
    standard_data = sorted(standard['data'], key=lambda x: x['name'])
    test_data = sorted(df_dict, key=lambda x: x['name'])
    
    # Compare data structure
    assert len(test_data) == len(standard_data), "Number of products should match"
    
    # Compare each product's data
    for test_item, standard_item in zip(test_data, standard_data):
        assert test_item['name'] == standard_item['name'], f"Product names should match: {test_item['name']} vs {standard_item['name']}"
        assert np.isclose(test_item['subtotal_sales'], standard_item['subtotal_sales'], rtol=1e-3), \
            f"Sales for {test_item['name']} should match: {test_item['subtotal_sales']} vs {standard_item['subtotal_sales']}"
    
    # Compare visualization type
    assert viz['type'] == standard['visualization']['type'], "Visualization type should be pie chart"
    assert 'plot' in viz, "Visualization should include plot data"
    
    # Save current results for comparison
    result = {
        'data': df_dict,  # This matches the standard format exactly
        'visualization': {
            'type': viz['type'],
            'plot': viz['plot'],
            'reason': viz.get('reason', '')
        }
    }
    
    with open('tests/logs/product_sales.json', 'w') as f:
        json.dump(result, f, indent=2)

# def test_category_product_count(ai_agent):
#     """Test counting products sold in each category"""
#     query = "How many products in total?"
    
#     # Generate and execute SQL query
#     sql_query = ai_agent.generate_sql_query(query)
#     df = ai_agent.execute_query(sql_query)
    
#     # Validate DataFrame structure
#     assert 'category' in df.columns, "DataFrame should contain category"
#     assert len(df) > 0, "DataFrame should not be empty"
    
#     # Save results
#     with open('tests/category_sales.json', 'w') as f:
#         json.dump(df.to_dict('records'), f, indent=2)

# def test_distinct_products(ai_agent):
#     """Test counting distinct products"""
#     query = "How many kinds of products in each category?"
    
#     # Generate and execute SQL query
#     sql_query = ai_agent.generate_sql_query(query)
#     df = ai_agent.execute_query(sql_query)
    
#     # Get the count from the DataFrame
#     product_count = df.iloc[0, 0] if len(df) > 0 else 0
    
#     # Validate result
#     assert isinstance(product_count, (int, float)), "Result should be a number"
#     assert product_count > 0, "Should have at least one product"
    
#     # Save result
#     with open('tests/product_count.json', 'w') as f:
#         json.dump({'distinct_products': int(product_count)}, f, indent=2)

if __name__ == "__main__":
    # Create agent instance
    agent = AIDBAgent(model="ollama")
    agent.connect_to_db(DATABASE_URL)
    
    # Run tests
    test_product_sales_pie_chart(agent)
    # test_category_product_count(agent)
    # test_distinct_products(agent) 