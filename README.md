# Essay Grading System

A simple AI-powered essay grading system that uses OpenAI's GPT models to automatically score essays. The system supports reading essays from CSV files and writing results to Google Sheets.

## Features

- **AI-Powered Grading**: Uses OpenAI GPT models for essay evaluation
- **Configurable Rubrics**: JSON-based rubric system for grading criteria
- **CSV Input**: Process essays from CSV files
- **Google Sheets Integration**: Write results directly to Google Sheets
- **Batch Processing**: Handle multiple essays efficiently
- **Concurrent Processing**: Grade multiple essays simultaneously for faster throughput

## Concurrency & Performance

**Concurrent Processing:**
- Time ≈ (n / w) × t_avg + overhead
- Where w = number of workers (concurrency level)

**Additional Considerations:**
- **API Rate Limits**: OpenAI has rate limits (RPM/TPM).
- **Criteria Splitting**: If using multiple criteria (m), each essay may require m API calls
- **Total API Calls**: n submissions × m criteria × workflow complexity
- **Memory Usage**: No constraints

Set `max_workers` in `scripts/grade_essays.py` to configure


## Project Structure

```
essay-grading/
├── config/
│   ├── settings.py          # System settings and API configuration
│   └── rubric.json         # Grading rubric definition
├── src/
│   ├── ai_agent/           # AI grading logic
│   │   ├── simple_grader.py    # Main grading class
│   │   └── openai_client.py    # OpenAI API client
│   ├── models/             # Data models
│   ├── sheets_integration/ # Google Sheets functionality
│   └── utils/              # Utility functions
├── scripts/
│   └── grade_essays.py     # Main grading script
└── requirements.txt        # Dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
SHEETS_CREDENTIALS_BASE64=your_base64_encoded_credentials
```


## Usage

### Quick Start

```bash
python scripts/grade_essays.py
```

### CSV Format

Your CSV file should have these columns:
- `essay_id`: Unique identifier for each essay
- `full_text`: The essay content to be graded

### Python API

```python
from src.ai_agent.simple_grader import SimpleGrader
from src.models.essay import create_essay

# Initialize grader
grader = SimpleGrader()

# Create and grade an essay
essay = create_essay(
    essay_id="test_001", 
    text="Your essay text here"
)
score = grader.grade_essay(essay)

print(f"Score: {score.total_score}")
print(f"Reasoning: {score.reasoning}")
```

## Configuration

### System Settings

Modify `config/settings.py` for system configuration:

```python
# Model settings
DEFAULT_MODEL = "gpt-4"
MODEL_TEMPERATURE = 0.3
MAX_TOKENS = 1500

# Processing settings  
BATCH_SIZE = 10
LOG_LEVEL = "INFO"
```

## Output

Results written to Google Sheets include:
- Essay ID and final score
- Detailed reasoning for the grade
- Timestamp of grading
- Individual criteria scores (if configured)

## Troubleshooting

### Debugging

Enable debug logging by setting `LOG_LEVEL = "DEBUG"` in `config/settings.py`.
