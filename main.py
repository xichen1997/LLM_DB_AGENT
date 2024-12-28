from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect
import pandas as pd
import plotly.express as px
import json
from typing import Optional, Dict, Any
from openai import OpenAI
from datetime import datetime
from jinja2 import Template
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()  # Add this near the top of your file

class QueryRequest(BaseModel):
    query: str
    db_connection: str

class AIDBAgent:
    def __init__(self):
        # Get API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=openai_api_key)
        self.engine = None
        self.db_schema = None

    def connect_to_db(self, connection_string: str):
        """Connect to database and extract schema information"""
        try:
            self.engine = create_engine(connection_string)
            inspector = inspect(self.engine)
            
            # Extract database schema
            self.db_schema = {}
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                self.db_schema[table_name] = {
                    'columns': [{'name': col['name'], 'type': str(col['type'])} 
                              for col in columns]
                }
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    def generate_sql_query(self, natural_query: str) -> str:
        """Convert natural language to SQL using OpenAI"""
        try:
            prompt = f"""
            Given the following database schema:
            {json.dumps(self.db_schema, indent=2)}
            
            Convert this natural language query to SQL:
            "{natural_query}"
            
            Return only the raw SQL query without any markdown formatting, explanation, or backticks.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a SQL query generator. Return only raw SQL without any markdown formatting or backticks."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Clean up the response by removing any markdown SQL formatting
            sql_query = response.choices[0].message.content.strip()
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            return sql_query
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query generation error: {str(e)}")

    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            return pd.read_sql_query(sql_query, self.engine)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")

    def generate_visualization(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Generate appropriate visualization based on data"""
        try:
            prompt = f"""
            Given this data summary:
            Columns: {df.columns.tolist()}
            Data types: {df.dtypes.to_dict()}
            Query: "{query}"
            
            Suggest the best visualization type (pie, bar, line, scatter, or table) and explain why.
            Return in JSON format with keys: 'type', 'x_column', 'y_column' (if applicable), 'reason'
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            viz_suggestion = json.loads(response.choices[0].message.content)
            
            # Create visualization
            if viz_suggestion['type'] == 'table':
                return {
                    'type': 'table',
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist()
                }
            
            # Create Plotly figure with improved formatting
            if viz_suggestion['type'] == 'bar':
                fig = px.bar(df, 
                    x=viz_suggestion['x_column'], 
                    y=viz_suggestion['y_column'],
                    title='Product Revenue Comparison',
                    labels={
                        viz_suggestion['x_column']: 'Product',
                        viz_suggestion['y_column']: 'Revenue ($)'
                    },
                    color_discrete_sequence=['#2c3e50'])
                
                # Update layout for better formatting
                fig.update_layout(
                    yaxis_tickformat='$,.2f',
                    hovermode='x',
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Update hover template
                fig.update_traces(
                    hovertemplate="Product: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>"
                )

            # Similar improvements for other chart types...
            
            return {
                'type': viz_suggestion['type'],
                'plot': fig.to_json(),
                'reason': viz_suggestion['reason'],
                'data': df.to_dict('records'),  # Add raw data for table view
                'columns': df.columns.tolist()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

    def generate_summary(self, df: pd.DataFrame, query: str) -> str:
        """Generate natural language summary of the results"""
        try:
            prompt = f"""
            Given this data summary:
            Number of rows: {len(df)}
            Columns: {df.columns.tolist()}
            Summary statistics: {df.describe().to_dict()}
            Query: "{query}"
            
            Provide a brief, natural language summary of the key insights from this data.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data analyst providing clear, concise summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summary generation error: {str(e)}")

    def generate_report(self, query: str, sql_query: str, visualization: Dict[str, Any], summary: str) -> str:
        """Generate an HTML report with the query results"""
        try:
            # Create reports directory if it doesn't exist
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Create a timestamp for the report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.html"
            
            # Improved HTML template with better styling and structure
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Database Query Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; 
                        margin: 40px;
                        line-height: 1.6;
                        color: #333;
                        background-color: #f5f7fa;
                    }
                    .container { 
                        max-width: 1200px; 
                        margin: 0 auto;
                        padding: 20px;
                    }
                    .section { 
                        margin-bottom: 40px;
                        background: white;
                        padding: 25px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .query-box { 
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 6px;
                        border: 1px solid #e9ecef;
                    }
                    .timestamp {
                        color: #666;
                        font-size: 0.9em;
                        margin-top: -20px;
                        margin-bottom: 30px;
                    }
                    table { 
                        border-collapse: collapse; 
                        width: 100%;
                        margin: 20px 0;
                        background: white;
                    }
                    th, td { 
                        border: 1px solid #dee2e6;
                        padding: 12px;
                        text-align: left;
                    }
                    th { 
                        background-color: #f8f9fa;
                        font-weight: 600;
                    }
                    h1, h2 { 
                        color: #2c3e50; 
                        margin-top: 0;
                    }
                    pre {
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        background-color: #f8f9fa;
                        padding: 15px;
                        border-radius: 4px;
                        border: 1px solid #e9ecef;
                        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace;
                        font-size: 14px;
                    }
                    .plot-container {
                        margin: 20px 0;
                        background: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }
                    .visualization-note {
                        margin-top: 20px;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 6px;
                        border-left: 4px solid #2c3e50;
                    }
                    .data-table {
                        margin-top: 30px;
                        border: 1px solid #e9ecef;
                        border-radius: 8px;
                        overflow: hidden;
                    }
                    .data-table h3 {
                        margin: 0;
                        padding: 15px 20px;
                        background: #f8f9fa;
                        border-bottom: 1px solid #e9ecef;
                    }
                    .table-container {
                        padding: 20px;
                        overflow-x: auto;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Database Query Report</h1>
                    <div class="timestamp">Generated on: {{ timestamp }}</div>
                    
                    <div class="section">
                        <h2>Query</h2>
                        <div class="query-box">
                            <p><strong>Natural Language Query:</strong> {{ natural_query }}</p>
                            <p><strong>SQL Query:</strong><br>
                            <pre>{{ sql_query }}</pre></p>
                        </div>
                    </div>

                    <div class="section">
                        <h2>Summary</h2>
                        <p>{{ summary }}</p>
                    </div>

                    <div class="section">
                        <h2>Visualization</h2>
                        {% if visualization.type != 'table' %}
                            <div id="plot" class="plot-container"></div>
                            <div class="visualization-note">
                                <strong>Visualization Choice:</strong> {{ visualization.reason }}
                            </div>
                            <script>
                                var plotData = {{ visualization.plot | safe }};
                                Plotly.newPlot('plot', plotData.data, plotData.layout);
                            </script>
                            
                            <div class="data-table">
                                <h3>Raw Data</h3>
                                <div class="table-container">
                                    <table>
                                        <thead>
                                            <tr>
                                            {% for column in visualization.columns %}
                                                <th>{{ column | title }}</th>
                                            {% endfor %}
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for row in visualization.data %}
                                            <tr>
                                                {% for column in visualization.columns %}
                                                <td>{{ row[column] }}</td>
                                                {% endfor %}
                                            </tr>
                                            {% endfor %}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        {% else %}
                            <table>
                                <thead>
                                    <tr>
                                    {% for column in visualization.columns %}
                                        <th>{{ column }}</th>
                                    {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for row in visualization.data %}
                                    <tr>
                                        {% for column in visualization.columns %}
                                        <td>{{ row[column] }}</td>
                                        {% endfor %}
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        {% endif %}
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Render the template with current timestamp
            template = Template(template_str)
            html_content = template.render(
                natural_query=query,
                sql_query=sql_query,
                summary=summary,
                visualization=visualization,
                timestamp=datetime.now().strftime("%B %d, %Y %H:%M:%S")
            )
            
            # Save the report
            report_path = reports_dir / report_filename
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content.strip())  # Remove leading/trailing whitespace
                
            return str(report_path)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")

app = FastAPI()
try:
    ai_agent = AIDBAgent()
except ValueError as e:
    print(f"Error initializing AIDBAgent: {e}")
    print("Please set the OPENAI_API_KEY environment variable")
    raise

@app.post("/query")
async def process_query(request: QueryRequest):
    # Connect to database
    ai_agent.connect_to_db(request.db_connection)
    
    # Generate and execute SQL query
    sql_query = ai_agent.generate_sql_query(request.query)
    results_df = ai_agent.execute_query(sql_query)
    
    # Generate visualization and summary
    visualization = ai_agent.generate_visualization(results_df, request.query)
    summary = ai_agent.generate_summary(results_df, request.query)
    
    # Generate report
    report_path = ai_agent.generate_report(request.query, sql_query, visualization, summary)
    
    return {
        "report_path": report_path,
        "sql_query": sql_query,
        # "visualization": visualization,
        "summary": summary
    }
