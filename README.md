# Dash Application

A web application built with Dash for data visualization and analysis, featuring OpenAI and Tavily API integration.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

1. Copy the example environment file:

   ```bash
   cp env.example .env
   ```

2. Edit `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   DEBUG=False
   ```

### 3. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:8050`

## Required API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── env.example        # Environment variables template
├── .env               # Your environment variables (create this)
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Development

- Set `DEBUG=True` in your `.env` file for development mode
- The application includes basic error handling for missing API keys
- All sensitive files are excluded from version control via `.gitignore`

## Troubleshooting

If you encounter API key errors:

1. Ensure your `.env` file exists and contains valid API keys
2. Check that the API keys are active and have sufficient credits
3. Verify the environment variables are loaded correctly by checking the console output
