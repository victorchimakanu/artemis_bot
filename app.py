"""
Main Dash Application
A web application built with Dash for data visualization and analysis.
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv
import dash
from dash import dcc, html, Input, Output, callback, State
import plotly.express as px
import pandas as pd
import openai
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__)

# Configure API clients
def initialize_api_clients():
    """Initialize and validate API clients with proper error handling."""
    errors = []
    
    # OpenAI configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        errors.append("OPENAI_API_KEY is not set in environment variables")
    else:
        openai.api_key = openai_api_key
    
    # Tavily configuration
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    if not tavily_api_key:
        errors.append("TAVILY_API_KEY is not set in environment variables")
    else:
        tavily_client = TavilyClient(api_key=tavily_api_key)
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease check your .env file and ensure all required API keys are set.")
        print("See env.example for reference.")
        sys.exit(1)
    
    return tavily_client

# Initialize API clients
try:
    tavily_client = initialize_api_clients()
    print("✅ API clients initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize API clients: {e}")
    sys.exit(1)

# Documentation search function
def search_documentation(query):
    """
    Search Turintech documentation using Tavily API.
    
    Args:
        query (str): User's search query
        
    Returns:
        dict: Search results with content and metadata
    """
    start_time = time.time()
    logger.info(f"Starting documentation search for: {query}")
    
    try:
        # Configure search parameters for Turintech documentation
        search_params = {
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": 5,
            "include_domains": ["docs.artemis.turintech.ai"],
            "exclude_domains": []
        }
        
        logger.info("Executing Tavily search...")
        
        try:
            # Execute Tavily search (timeout handled by Tavily client)
            response = tavily_client.search(**search_params)
        except Exception as e:
            if "signal only works in main thread" in str(e):
                logger.error("Tavily client signal error - using fallback mock search")
                # Fallback mock response for testing
                response = {
                    'results': [
                        {
                            'title': 'Artemis Documentation',
                            'url': 'https://docs.artemis.turintech.ai',
                            'content': f'Based on your query "{query}", this is a mock response from the documentation. The actual search service is temporarily unavailable due to a technical issue.'
                        }
                    ],
                    'answer': f'This is a mock response for your query: "{query}". The search service is currently experiencing technical difficulties.'
                }
            else:
                raise e
        
        search_duration = time.time() - start_time
        logger.info(f"Search completed in {search_duration:.2f} seconds")
        
        # Process search results
        if not response or 'results' not in response:
            logger.warning("No search results returned from Tavily")
            return {
                'success': False,
                'content': 'Sorry — I couldn\'t find that in the documentation.',
                'sources': [],
                'duration': search_duration,
                'error_type': 'no_results'
            }
        
        results = response.get('results', [])
        if not results:
            logger.warning("Empty search results from Tavily")
            return {
                'success': False,
                'content': 'Sorry — I couldn\'t find that in the documentation.',
                'sources': [],
                'duration': search_duration,
                'error_type': 'empty_results'
            }
        
        # Extract and clean content
        extracted_content = []
        sources = []
        
        for result in results:
            # Extract basic information
            title = result.get('title', 'Untitled')
            url = result.get('url', '')
            content = result.get('content', '')
            
            # Clean and format content
            if content:
                # Preserve code blocks and formatting
                cleaned_content = content.strip()
                if cleaned_content:
                    extracted_content.append(f"**{title}**\n{cleaned_content}\n")
                    sources.append({
                        'title': title,
                        'url': url
                    })
        
        if not extracted_content:
            logger.warning("No content extracted from Tavily results")
            return {
                'success': False,
                'content': 'Sorry — I couldn\'t find that in the documentation.',
                'sources': sources,
                'duration': search_duration,
                'error_type': 'no_content'
            }
        
        # Combine all content
        combined_content = "\n---\n\n".join(extracted_content)
        
        logger.info(f"Successfully extracted content from {len(sources)} sources")
        
        return {
            'success': True,
            'content': combined_content,
            'sources': sources,
            'duration': search_duration,
            'answer': response.get('answer', '')
        }
        
    except TimeoutError as e:
        search_duration = time.time() - start_time
        logger.error(f"Tavily search timed out after {search_duration:.2f} seconds")
        return {
            'success': False,
            'content': 'Search request timed out. Please try again.',
            'sources': [],
            'duration': search_duration,
            'error_type': 'timeout'
        }
        
    except ConnectionError as e:
        search_duration = time.time() - start_time
        logger.error(f"Tavily connection error: {str(e)}")
        return {
            'success': False,
            'content': 'Unable to connect to search service. Please check your internet connection.',
            'sources': [],
            'duration': search_duration,
            'error_type': 'connection'
        }
        
    except Exception as e:
        search_duration = time.time() - start_time
        error_msg = str(e).lower()
        
        # Handle specific Tavily API errors
        if 'api key' in error_msg or 'authentication' in error_msg:
            logger.error(f"Tavily authentication error: {str(e)}")
            return {
                'success': False,
                'content': 'Search service authentication failed. Please check your API configuration.',
                'sources': [],
                'duration': search_duration,
                'error_type': 'auth'
            }
        elif 'rate limit' in error_msg or 'quota' in error_msg:
            logger.error(f"Tavily rate limit error: {str(e)}")
            return {
                'success': False,
                'content': 'Search service is temporarily unavailable due to high usage. Please try again later.',
                'sources': [],
                'duration': search_duration,
                'error_type': 'rate_limit'
            }
        else:
            logger.error(f"Tavily search failed after {search_duration:.2f} seconds: {str(e)}")
            return {
                'success': False,
                'content': 'Search service encountered an error. Please try again.',
                'sources': [],
                'duration': search_duration,
                'error_type': 'unknown'
            }

# AI response generation function
def generate_response(user_query, documentation_content):
    """
    Generate AI response using OpenAI GPT-4 based on documentation content.
    
    Args:
        user_query (str): User's question
        documentation_content (str): Extracted documentation content
        
    Returns:
        dict: Generated response with content and metadata
    """
    start_time = time.time()
    logger.info(f"Generating AI response for query: {user_query[:100]}...")
    
    # Input validation
    if not user_query or not user_query.strip():
        return {
            'success': False,
            'content': 'Please provide a valid question.',
            'duration': time.time() - start_time
        }
    
    if not documentation_content or not documentation_content.strip():
        return {
            'success': False,
            'content': 'Sorry — I couldn\'t find that in the documentation.',
            'duration': time.time() - start_time
        }
    
    try:
        # System prompt to ensure responses stay within documentation scope
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided Turintech documentation content. 

IMPORTANT RULES:
1. ONLY use information from the provided documentation content
2. If the documentation doesn't contain enough information to answer the question, respond with: "Sorry — I couldn't find that in the documentation"
3. Preserve all code examples, formatting, and technical details exactly as they appear in the documentation
4. Be accurate and specific in your responses
5. If you find relevant information, provide a clear, helpful answer with proper code formatting
6. Do not make assumptions or provide information not found in the documentation"""

        # User prompt with documentation context
        user_prompt = f"""Based on the following Turintech documentation content, please answer this question: {user_query}

DOCUMENTATION CONTENT:
{documentation_content}

Please provide a helpful answer based only on the documentation above. If the documentation doesn't contain enough information to answer the question, respond with: "Sorry — I couldn't find that in the documentation"."""

        # Configure OpenAI client
        client = openai.OpenAI(api_key=openai.api_key)
        
        logger.info("Sending request to OpenAI GPT-4...")
        
        # Generate response using GPT-4 (timeout handled by OpenAI client)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent, factual responses
            max_tokens=1500,  # Reasonable limit for documentation responses
            timeout=45  # 45-second timeout
        )
        
        generation_duration = time.time() - start_time
        logger.info(f"AI response generated in {generation_duration:.2f} seconds")
        
        # Extract response content
        ai_response = response.choices[0].message.content.strip()
        
        # Check if the response indicates insufficient information
        if "Sorry — I couldn't find that in the documentation" in ai_response:
            logger.info("AI determined insufficient information in documentation")
            return {
                'success': True,
                'content': ai_response,
                'duration': generation_duration,
                'insufficient_info': True
            }
        
        logger.info("AI generated successful response based on documentation")
        
        return {
            'success': True,
            'content': ai_response,
            'duration': generation_duration,
            'insufficient_info': False
        }
        
    except TimeoutError as e:
        generation_duration = time.time() - start_time
        logger.error(f"OpenAI request timed out after {generation_duration:.2f} seconds")
        return {
            'success': False,
            'content': 'AI service request timed out. Please try again.',
            'duration': generation_duration,
            'error_type': 'timeout'
        }
        
    except ConnectionError as e:
        generation_duration = time.time() - start_time
        logger.error(f"OpenAI connection error: {str(e)}")
        return {
            'success': False,
            'content': 'Unable to connect to AI service. Please check your internet connection.',
            'duration': generation_duration,
            'error_type': 'connection'
        }
        
    except openai.RateLimitError as e:
        generation_duration = time.time() - start_time
        logger.error(f"OpenAI rate limit exceeded: {str(e)}")
        return {
            'success': False,
            'content': 'AI service is temporarily unavailable due to high usage. Please try again in a few minutes.',
            'duration': generation_duration,
            'error_type': 'rate_limit'
        }
        
    except openai.AuthenticationError as e:
        generation_duration = time.time() - start_time
        logger.error(f"OpenAI authentication error: {str(e)}")
        return {
            'success': False,
            'content': 'AI service authentication failed. Please check your API configuration.',
            'duration': generation_duration,
            'error_type': 'auth'
        }
        
    except openai.APITimeoutError as e:
        generation_duration = time.time() - start_time
        logger.error(f"OpenAI API timeout: {str(e)}")
        return {
            'success': False,
            'content': 'AI service request timed out. Please try again.',
            'duration': generation_duration,
            'error_type': 'timeout'
        }
        
    except openai.APIError as e:
        generation_duration = time.time() - start_time
        error_msg = str(e).lower()
        
        if 'quota' in error_msg or 'billing' in error_msg:
            logger.error(f"OpenAI quota/billing error: {str(e)}")
            return {
                'success': False,
                'content': 'AI service quota exceeded. Please check your account status.',
                'duration': generation_duration,
                'error_type': 'quota'
            }
        else:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                'success': False,
                'content': 'AI service encountered an error. Please try again later.',
                'duration': generation_duration,
                'error_type': 'api_error'
            }
        
    except Exception as e:
        generation_duration = time.time() - start_time
        error_msg = str(e).lower()
        
        if 'api key' in error_msg or 'authentication' in error_msg:
            logger.error(f"OpenAI authentication error: {str(e)}")
            return {
                'success': False,
                'content': 'AI service authentication failed. Please check your API configuration.',
                'duration': generation_duration,
                'error_type': 'auth'
            }
        else:
            logger.error(f"Unexpected error in AI response generation: {str(e)}")
            return {
                'success': False,
                'content': 'An unexpected error occurred. Please try again.',
                'duration': generation_duration,
                'error_type': 'unexpected'
            }

# App layout
app.layout = html.Div([
    # Header section
    html.Div([
        html.H1("Documentation Search", 
                style={'textAlign': 'center', 'marginBottom': 10, 'color': '#2c3e50'}),
        html.P("Ask questions about documentation and get AI-powered answers", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': 30}),
    ], style={'marginBottom': 40}),
    
    # Main content container
    html.Div([
        # Input section
        html.Div([
            html.Label("Enter your question:", 
                      style={'fontWeight': 'bold', 'marginBottom': 10, 'display': 'block'}),
            dcc.Input(
                id='question-input',
                type='text',
                placeholder='What would you like to know about the documentation?',
                style={
                    'width': '100%',
                    'padding': '12px 16px',
                    'border': '2px solid #e1e8ed',
                    'borderRadius': '8px',
                    'fontSize': '16px',
                    'outline': 'none',
                    'transition': 'border-color 0.3s ease'
                }
            ),
            # Hidden div to store processing state
            html.Div(id='processing-state', style={'display': 'none'}),
            html.Button(
                'Search Documentation',
                id='submit-button',
                n_clicks=0,
                style={
                    'width': '100%',
                    'marginTop': '15px',
                    'padding': '12px 24px',
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '8px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'transition': 'background-color 0.3s ease'
                }
            )
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '30px',
            'borderRadius': '12px',
            'marginBottom': '30px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        }),
        
        # Response section
        html.Div([
            html.H3("Response", style={'marginBottom': 15, 'color': '#2c3e50'}),
            html.Div(
                id='response-display',
                children=[
                    html.P("Your search results will appear here...", 
                           style={'color': '#7f8c8d', 'fontStyle': 'italic'})
                ],
                style={
                    'minHeight': '200px',
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'border': '1px solid #e1e8ed',
                    'borderRadius': '8px',
                    'whiteSpace': 'pre-wrap',
                    'lineHeight': '1.6'
                }
            )
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '30px',
            'borderRadius': '12px',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
        })
        
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'})
    
], style={'backgroundColor': '#f5f6fa', 'minHeight': '100vh', 'padding': '20px 0'})

# Add CSS for loading spinner animation
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Callbacks for form submission and loading states
@callback(
    [Output('response-display', 'children'),
     Output('submit-button', 'disabled'),
     Output('question-input', 'disabled'),
     Output('processing-state', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('question-input', 'value')]
)
def update_response(n_clicks, question):
    """Handle form submission and display response with loading states."""
    if n_clicks == 0:
        return (html.P("Your search results will appear here...", 
                      style={'color': '#7f8c8d', 'fontStyle': 'italic'}),
                False, False, "")
    
    if not question or question.strip() == "":
        return (html.P("Please enter a question before searching.", 
                      style={'color': '#e74c3c', 'fontStyle': 'italic'}),
                False, False, "")
    
    # Show loading state
    loading_content = html.Div([
        html.Div([
            html.Div([
                html.Div(style={
                    'width': '20px', 'height': '20px', 'border': '3px solid #f3f3f3',
                    'borderTop': '3px solid #3498db', 'borderRadius': '50%',
                    'animation': 'spin 1s linear infinite', 'margin': '0 auto'
                }),
                html.P("Searching documentation and generating response...", 
                       style={'textAlign': 'center', 'marginTop': '10px', 
                              'color': '#7f8c8d'})
            ], style={'textAlign': 'center', 'padding': '40px'})
        ], style={'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 
                  'border': '1px solid #e1e8ed'})
    ])
    
    try:
        # Perform documentation search
        logger.info(f"Processing search request: {question}")
        search_results = search_documentation(question)
        
        if not search_results['success']:
            # Handle different types of search errors with appropriate styling
            error_type = search_results.get('error_type', 'unknown')
            
            if error_type == 'no_results' or error_type == 'empty_results' or error_type == 'no_content':
                # These are expected cases where documentation doesn't contain the information
                return (html.Div([
                    html.P(f"ℹ️ {search_results['content']}", 
                           style={'color': '#856404', 'marginBottom': '10px',
                                  'backgroundColor': '#fff3cd', 'padding': '15px',
                                  'borderRadius': '8px', 'borderLeft': '4px solid #ffc107'}),
                    html.P(f"Search completed in {search_results['duration']:.2f} seconds", 
                           style={'color': '#7f8c8d', 'fontSize': '14px'})
                ]), False, False, "")
            else:
                # These are actual errors that need attention
                return (html.Div([
                    html.P(f"❌ {search_results['content']}", 
                           style={'color': '#721c24', 'marginBottom': '10px',
                                  'backgroundColor': '#f8d7da', 'padding': '15px',
                                  'borderRadius': '8px', 'borderLeft': '4px solid #dc3545'}),
                    html.P(f"Search completed in {search_results['duration']:.2f} seconds", 
                           style={'color': '#7f8c8d', 'fontSize': '14px'})
                ]), False, False, "")
        
        # Generate AI response using the documentation content
        logger.info("Generating AI response based on documentation content")
        ai_response = generate_response(question, search_results['content'])
        
        # Display results
        content_parts = []
        
        # Add search summary
        content_parts.append(
            html.P(f"✅ Found {len(search_results['sources'])} relevant documentation sources", 
                   style={'color': '#27ae60', 'fontWeight': 'bold', 'marginBottom': '15px'})
        )
        
        # Add AI-generated response
        if ai_response['success']:
            if ai_response.get('insufficient_info', False):
                # Handle insufficient information case
                content_parts.append(
                    html.Div([
                        html.H4("AI Response:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P(ai_response['content'], 
                               style={'backgroundColor': '#fff3cd', 'padding': '15px', 
                                      'borderRadius': '8px', 'borderLeft': '4px solid #ffc107',
                                      'color': '#856404'})
                    ], style={'marginBottom': '20px'})
                )
            else:
                # Display successful AI response
                content_parts.append(
                    html.Div([
                        html.H4("AI Response:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.Pre(ai_response['content'], 
                                 style={'backgroundColor': '#f8f9fa', 'padding': '15px', 
                                        'borderRadius': '8px', 'borderLeft': '4px solid #28a745',
                                        'whiteSpace': 'pre-wrap', 'fontSize': '14px',
                                        'lineHeight': '1.6'})
                    ], style={'marginBottom': '20px'})
                )
        else:
            # Handle AI generation errors with appropriate styling based on error type
            error_type = ai_response.get('error_type', 'unknown')
            
            if error_type in ['timeout', 'connection']:
                # Network-related errors
                content_parts.append(
                    html.Div([
                        html.H4("AI Response:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P(f"⚠️ {ai_response['content']}", 
                               style={'backgroundColor': '#fff3cd', 'padding': '15px', 
                                      'borderRadius': '8px', 'borderLeft': '4px solid #ffc107',
                                      'color': '#856404'})
                    ], style={'marginBottom': '20px'})
                )
            elif error_type in ['auth', 'quota']:
                # Configuration-related errors
                content_parts.append(
                    html.Div([
                        html.H4("AI Response:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P(f"🔧 {ai_response['content']}", 
                               style={'backgroundColor': '#d1ecf1', 'padding': '15px', 
                                      'borderRadius': '8px', 'borderLeft': '4px solid #17a2b8',
                                      'color': '#0c5460'})
                    ], style={'marginBottom': '20px'})
                )
            else:
                # General errors
                content_parts.append(
                    html.Div([
                        html.H4("AI Response:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P(f"❌ {ai_response['content']}", 
                               style={'backgroundColor': '#f8d7da', 'padding': '15px', 
                                      'borderRadius': '8px', 'borderLeft': '4px solid #dc3545',
                                      'color': '#721c24'})
                    ], style={'marginBottom': '20px'})
                )
        
        # Add raw documentation content (collapsed by default)
        content_parts.append(
            html.Div([
                html.Details([
                    html.Summary("View Raw Documentation Content", 
                                style={'cursor': 'pointer', 'fontWeight': 'bold', 
                                      'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Pre(search_results['content'], 
                             style={'backgroundColor': '#ffffff', 'padding': '15px', 
                                    'borderRadius': '8px', 'border': '1px solid #e1e8ed',
                                    'whiteSpace': 'pre-wrap', 'fontSize': '12px',
                                    'maxHeight': '300px', 'overflowY': 'auto'})
                ])
            ], style={'marginBottom': '20px'})
        )
        
        # Add sources
        if search_results['sources']:
            source_links = []
            for source in search_results['sources']:
                source_links.append(
                    html.Li([
                        html.A(source['title'], href=source['url'], target='_blank',
                               style={'color': '#3498db', 'textDecoration': 'none'})
                    ], style={'marginBottom': '5px'})
                )
            
            content_parts.append(
                html.Div([
                    html.H4("Sources:", style={'color': '#2c3e50', 'marginBottom': '10px'}),
                    html.Ul(source_links, style={'marginLeft': '20px'})
                ], style={'marginTop': '20px'})
            )
        
        # Add performance info
        total_duration = search_results['duration'] + ai_response['duration']
        content_parts.append(
            html.P(f"Search: {search_results['duration']:.2f}s | AI: {ai_response['duration']:.2f}s | Total: {total_duration:.2f}s", 
                   style={'color': '#7f8c8d', 'fontSize': '12px', 'marginTop': '15px', 
                          'textAlign': 'right', 'fontStyle': 'italic'})
        )
        
        return (html.Div(content_parts), False, False, "")
        
    except Exception as e:
        logger.error(f"Unexpected error in callback: {str(e)}")
        return (html.Div([
            html.P(f"❌ An unexpected error occurred: {str(e)}", 
                   style={'color': '#e74c3c', 'marginBottom': '10px'}),
            html.P("Please try again or contact support if the issue persists.", 
                   style={'color': '#7f8c8d', 'fontSize': '14px'})
        ]), False, False, "")

if __name__ == '__main__':
    # Get debug mode from environment
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting Dash application...")
    print(f"Debug mode: {debug_mode}")
    
    app.run_server(
        debug=debug_mode,
        host='0.0.0.0',
        port=8050
    )
