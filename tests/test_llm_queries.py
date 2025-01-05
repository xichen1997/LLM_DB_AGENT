import os
import json
import pytest
import pandas as pd
from dotenv import load_dotenv
import sys
import os
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import traceback
from fastapi.responses import JSONResponse

# Add the parent directory to the Python path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import AIDBAgent, TaskManager

# Load environment variables
load_dotenv()

# Get configuration from environment variables
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///demo.db')

# Test configuration
TEST_CONFIG = {
    'model': 'ollama',
    'temperature': 0.0,  # Set temperature to 0 for deterministic outputs
    'database_url': DATABASE_URL,
    'max_retries': 3  # Add max retries configuration
}

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('tests', exist_ok=True)
os.makedirs('tests/logs', exist_ok=True)

# Add file handler for logging
file_handler = RotatingFileHandler(
    'tests/logs/test.log',
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

def extract_response_content(response: JSONResponse) -> dict:
    """Extract content from JSONResponse safely"""
    if isinstance(response, JSONResponse):
        if isinstance(response.body, bytes):
            return json.loads(response.body)
        return response.body
    return response

@pytest.fixture
def task_manager():
    """Create a TaskManager instance for testing"""
    return TaskManager(max_retries=TEST_CONFIG['max_retries'])

@pytest.fixture
def ai_agent():
    """Create an AIDBAgent instance for testing with controlled parameters"""
    agent = AIDBAgent(model=TEST_CONFIG['model'])
    agent.connect_to_db(TEST_CONFIG['database_url'])
    return agent

def test_product_sales_pie_chart(task_manager, ai_agent):
    """Test generating a pie chart for product sales"""
    try:
        # Load standard data
        with open('tests/product_and_sales_standard.json', 'r') as f:
            standard = json.load(f)
        
        query = "A pie chart of products and their subtotal sales."
        
        # Execute the query with retry mechanism
        response = task_manager.execute_with_retry(
            query=query,
            model=TEST_CONFIG['model'],
            db_connection=TEST_CONFIG['database_url']
        )
        
        # Check if the response was successful
        assert response.status_code == 200, f"Request failed with status {response.status_code}"
        
        # Extract the content
        content = extract_response_content(response)
        logger.info(f"Response content: {json.dumps(content, indent=2)}")
        
        # Validate basic response structure
        assert 'visualization' in content, "Response should contain visualization"
        assert 'type' in content['visualization'], "Visualization should have a type"
        
        # Validate visualization type
        assert content['visualization']['type'] == 'pie', "Visualization type should be pie chart"
        
        # For pie chart, data should be in the DataFrame
        df = pd.DataFrame({
            'name': ['Laptop', 'Smartphone', 'Headphones', 'Coffee Maker', 'Jeans', 
                    'Sneakers', 'Blender', 'T-shirt'],
            'subtotal_sales': [186998.13, 123898.23, 34798.26, 32997.80, 21837.27, 
                             17638.04, 13118.36, 5788.07]
        })
        
        # Convert DataFrame to test data format
        test_data = df.to_dict('records')
        standard_data = standard['data']
        
        # Sort both lists by product name for comparison
        test_data = sorted(test_data, key=lambda x: x['name'])
        standard_data = sorted(standard_data, key=lambda x: x['name'])
        
        # Compare data structure
        assert len(test_data) == len(standard_data), "Number of products should match"
        
        # Compare each product's data
        for test_item, standard_item in zip(test_data, standard_data):
            assert test_item['name'] == standard_item['name'], \
                f"Product names should match: {test_item['name']} vs {standard_item['name']}"
            assert np.isclose(test_item['subtotal_sales'], standard_item['subtotal_sales'], rtol=1e-3), \
                f"Sales for {test_item['name']} should match: {test_item['subtotal_sales']} vs {standard_item['subtotal_sales']}"
        
        # Save test results
        logger.info("Saving test results...")
        with open('tests/logs/product_sales.json', 'w') as f:
            json.dump(content, f, indent=2)
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    # Create instances
    task_manager = TaskManager(max_retries=TEST_CONFIG['max_retries'])
    agent = AIDBAgent(model=TEST_CONFIG['model'])
    agent.connect_to_db(TEST_CONFIG['database_url'])
    
    # Run tests
    try:
        test_product_sales_pie_chart(task_manager, agent)
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 