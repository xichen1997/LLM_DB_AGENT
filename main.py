from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, inspect
import pandas as pd
import plotly.express as px
import json
from typing import Optional, Dict, Any
import requests
from datetime import datetime
import os
import base64
import io
from fastapi.middleware.cors import CORSMiddleware
import logging
from logging.handlers import RotatingFileHandler
import traceback
from openai import OpenAI
from enum import Enum
import numpy as np
import random

# Set manual seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Add file handler for logging
file_handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

# Create directories for templates and static files
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

app = FastAPI()
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Setup templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    db_connection: str
    model: str  # 'ollama' or 'chatgpt'
    api_key: Optional[str] = None  # Optional API key for ChatGPT

class LLMProvider(Enum):
    OLLAMA = "ollama"
    CHATGPT = "chatgpt"

class AIDBAgent:
    def __init__(self, model: str = "ollama", api_key: Optional[str] = None):
        logger.info(f"Initializing AIDBAgent with {model}")
        try:
            self.model = model.lower()
            self.ollama_url = "http://localhost:11434/api/generate"
            self.openai_client = OpenAI(api_key=api_key) if api_key else None
            self.engine = None
            self.db_schema = None
            
            if self.model == "chatgpt" and not api_key:
                raise ValueError("API key is required for ChatGPT")
                
        except Exception as e:
            logger.error(f"Error initializing AIDBAgent: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Unified method to call LLM providers"""
        try:
            if self.model == "ollama":
                # Combine system and user prompts
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Prepare Ollama API parameters
                ollama_request = {
                    "model": "llama3.2:3b",
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.0,
                    # "stop": [";", "\n\n"],  # Add stop tokens
                    # "context": [],  # Reset context for each request
                    "seed": 42  # Add this to fix the random seed
                }
                
                response = requests.post(self.ollama_url, json=ollama_request)
                
                if response.status_code == 200:
                    return response.json()['response'].strip()
                else:
                    raise Exception(f"Ollama API error: {response.text}")
                    
            elif self.model == "chatgpt":
                if not self.openai_client:
                    raise ValueError("OpenAI client not initialized")
                    
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Fixed model name typo
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,  # Set temperature to 0 for deterministic output
                    seed=42,          # Set a fixed seed for reproducibility
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            logger.error(f"LLM call error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def connect_to_db(self, connection_string: str):
        """Connect to database and extract schema information"""
        try:
            logger.info(f"Connecting to database: {connection_string}")
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
            logger.info("Database connection and schema extraction successful")
            return True
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

    def generate_sql_query(self, natural_query: str) -> str:
        """Convert natural language to SQL using selected LLM provider"""
        try:
            logger.info(f"Generating SQL query for: {natural_query}")
            
            system_prompt = """You are a SQL expert specialized in converting natural language queries to SQL.
            Your responses should ONLY contain the SQL query, without any explanations or markdown.
            
            Follow these rules strictly:
            1. Always verify table and column names against the provided schema
            2. Use appropriate JOIN conditions when multiple tables are involved
            3. Include WHERE clauses to filter data as specified
            4. Use proper aggregation functions (SUM, AVG, COUNT, etc.) when needed
            5. Handle date/time operations correctly
            6. Ensure proper GROUP BY clauses when using aggregations
            7. Add HAVING clauses when filtering aggregated results
            8. Use ORDER BY for sorting when relevant
            9. Limit results when appropriate
            10. Use appropriate data type conversions if needed
            
            If you cannot generate a valid query, respond with "ERROR: " followed by the reason.
            """

            user_prompt = f"""Database Schema:
            {json.dumps(self.db_schema, indent=2)}
            
            Natural Language Query: "{natural_query}"

            Requirements:
            1. Generate a single, executable SQL query
            2. Only use tables and columns that exist in the schema
            3. Ensure all table joins are properly specified
            4. Include appropriate WHERE conditions
            5. Handle NULL values appropriately

            Generate the SQL query:"""

            sql_query = self._llm_call(system_prompt, user_prompt)
            
            # Remove any markdown formatting if present
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            # Check if the response indicates an error
            if sql_query.startswith('ERROR:'):
                raise Exception(sql_query)
            
            # Validate that the query contains basic SQL keywords
            if not any(keyword in sql_query.upper() for keyword in ['SELECT', 'FROM']):
                raise Exception("Generated query does not contain basic SQL syntax")
            
            logger.info(f"Generated SQL query: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Query generation error: {str(e)}")
            logger.error(traceback.format_exc())
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
            logger.info(f"Generating visualization for dataframe with columns: {df.columns.tolist()}")
            system_prompt = """You are a data visualization expert. Your task is to analyze the data and determine the most appropriate visualization type.
            Be explicit about choosing table visualization when the data needs detailed examination.
            The visualization type should comply with the users query.
            """
            
            user_prompt = f"""
            Given this data summary:
            Columns: {df.columns.tolist()}
            Data types: {df.dtypes.to_dict()}
            Number of rows: {len(df)}
            Query: "{query}"

            
            Return a JSON object with the following structure for visualization, without any markdown or code blocks:
            {{
                "type": "value" | "line" | "bar" | "scatter" | "pie" | "table",
                "reason": "explanation for choosing this visualization",
                "value_column": "column_name",  // for 'value' type only
                "names": "column_name",  // for 'pie' type only
                "values": "column_name",  // for 'pie' type only
                "x_column": "column_name",      // for pie/line/bar/scatter plots
                "y_column": "column_name",      // for pie/line/bar/scatter plots
                "title": "chart title"          // optional, for all chart types
            }}
            

            Choose visualization type based on these rules:
            1. For single aggregated values: use 'value' type
            2. For time series data: use 'line' type with sale_date as x_column
            3. For categorical comparisons: use 'bar' type
            4. For distributions: use 'scatter' type
            5. For parts of a whole: use 'pie' type
            6. For detailed data or if unsure: use 'table' type
            7. Dont add limit in the query until the user asks for it.
            
            If the user query mentions a specific type, then the visualization type should be that type.
            Return ONLY the JSON object, no additional text.
            without any markdown or code blocks.
            """
            
            content = self._llm_call(system_prompt, user_prompt)
            
            logger.info(f"Raw visualization suggestion: {content}")
            
            try:
                viz_suggestion = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {content}")
                # Fallback to table visualization if JSON parsing fails
                return {
                    'type': 'table',
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'reason': 'Fallback to table view due to visualization parsing error'
                }
            
            logger.info(f"Parsed visualization suggestion: {viz_suggestion}")
            
            # Update the visualization creation part
            if viz_suggestion['type'] == 'table':
                return {
                    'type': 'table',
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist(),
                    'reason': viz_suggestion.get('reason', 'Showing detailed data')
                }
            elif viz_suggestion['type'] == 'value':
                value_column = viz_suggestion.get('value_column', df.columns[0])
                return {
                    'type': 'value',
                    'data': float(df[value_column].iloc[0]) if len(df) > 0 else 0,
                    'label': value_column
                }
            else:
                try:
                    # Common parameters for all plot types
                    plot_params = {
                        'title': viz_suggestion.get('title', 'Data Visualization')
                    }

                    if viz_suggestion['type'] == 'pie':
                        values_col = viz_suggestion.get('values', viz_suggestion.get('y_column'))
                        names_col = viz_suggestion.get('names', viz_suggestion.get('x_column'))
                        
                        # Ensure values are numeric
                        if not np.issubdtype(df[values_col].dtype, np.number):
                            raise ValueError(f"Values column '{values_col}' must be numeric")
                        
                        # Aggregate data for pie chart
                        df_agg = df.groupby(names_col)[values_col].sum().reset_index()
                        fig = px.pie(df_agg, values=values_col, names=names_col, title=plot_params['title'])
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        
                    elif viz_suggestion['type'] in ['bar', 'line', 'scatter']:
                        x_col = viz_suggestion.get('x_column')
                        y_col = viz_suggestion.get('y_column')
                        
                        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                            raise KeyError(f"Missing or invalid columns for {viz_suggestion['type']} chart")
                        
                        # Create appropriate plot
                        if viz_suggestion['type'] == 'bar':
                            fig = px.bar(df, x=x_col, y=y_col, title=plot_params['title'])
                        elif viz_suggestion['type'] == 'line':
                            fig = px.line(df, x=x_col, y=y_col, title=plot_params['title'])
                        else:  # scatter
                            fig = px.scatter(df, x=x_col, y=y_col, title=plot_params['title'])
                        
                        # Update layout
                        fig.update_layout(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            showlegend=True
                        )
                    else:
                        # Fallback to table for unknown visualization type
                        logger.warning(f"Unknown visualization type: {viz_suggestion['type']}")
                        return {
                            'type': 'table',
                            'data': df.to_dict('records'),
                            'columns': df.columns.tolist(),
                            'reason': 'Fallback to table view due to unknown visualization type'
                        }

                    return {
                        'type': viz_suggestion['type'],
                        'plot': fig.to_json(),
                        'reason': viz_suggestion.get('reason', 'Data visualization')
                    }

                except Exception as e:
                    logger.error(f"Error creating visualization: {str(e)}")
                    logger.error(traceback.format_exc())
                    logger.error(f"DataFrame at error: {df}")
                    logger.error(f"Visualization suggestion at error: {viz_suggestion}")
                    # Fallback to table view
                    return {
                        'type': 'table',
                        'data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'reason': f'Fallback to table view due to error: {str(e)}'
                    }

        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            logger.error(traceback.format_exc())
            # Final fallback to table visualization
            return {
                'type': 'table',
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'reason': 'Fallback to table view due to error'
            }

    def generate_summary(self, df: pd.DataFrame, query: str) -> str:
        """Generate natural language summary of the results"""
        try:
            system_prompt = """You are a data analyst. Provide clear and concise summaries of data analysis results."""
            
            user_prompt = f"""
            Given this data summary:
            Number of rows: {len(df)}
            Columns: {df.columns.tolist()}
            Summary statistics: {df.describe().to_dict()}
            Query: "{query}"
            
            Provide a brief, natural language summary of the key insights from this data.
            """
            
            return self._llm_call(system_prompt, user_prompt)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summary generation error: {str(e)}")

    def save_report(self, query: str, sql_query: str, df: pd.DataFrame, visualization: Dict[str, Any], summary: str) -> str:
        """Save the report as a file"""
        try:
            # Create a unique filename
            filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
            
            # Create the HTML template
            template = """
            <html>
            <head>
                <title>Query Report</title>
            </head>
            <body>
                <h1>Query Report</h1>
                <h2>Query: {{ query }}</h2>
                <h3>SQL Query: {{ sql_query }}</h3>
                <h3>Visualization: {{ visualization['type'] }}</h3>
                <h3>Summary: {{ summary }}</h3>
                <h3>Data: {{ df.to_html() }}</h3>
            </body>
            </html>
            """
            
            # Render the template
            rendered_template = template.format(
                query=query,
                sql_query=sql_query,
                visualization=visualization,
                summary=summary,
                df=df.to_html()
            )
            
            # Save the rendered template to a file
            with open(filename, 'w') as f:
                f.write(rendered_template)
            
            return filename
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report saving error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/query")
async def process_query(request: QueryRequest):
    logger.info(f"Received query request: {request.query}")
    try:
        # Initialize with specified model and API key
        ai_agent = AIDBAgent(
            model=request.model,
            api_key=request.api_key
        )
        
        # Connect to database
        logger.info("Connecting to database...")
        ai_agent.connect_to_db(request.db_connection)
        
        # Generate and execute SQL query
        logger.info("Generating SQL query...")
        sql_query = ai_agent.generate_sql_query(request.query)
        logger.info("Executing SQL query...")
        results_df = ai_agent.execute_query(sql_query)
        
        # Generate visualization and summary
        logger.info("Generating visualization...")
        visualization = ai_agent.generate_visualization(results_df, request.query)
        logger.info("Generating summary...")
        summary = ai_agent.generate_summary(results_df, request.query)
        
        # Save report
        logger.info("Saving report...")
        report_path = ai_agent.save_report(
            query=request.query,
            sql_query=sql_query,
            df=results_df,
            visualization=visualization,
            summary=summary
        )
        
        logger.info("Request processed successfully")
        return JSONResponse(content={
            "sql_query": sql_query,
            "visualization": visualization,
            "summary": summary,
            "report_path": report_path
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=400,
            content={
                "error": str(e),
                "details": traceback.format_exc()
            }
        )
