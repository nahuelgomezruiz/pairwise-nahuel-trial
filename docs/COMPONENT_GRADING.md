# Component-Based Essay Grading System

## Overview

This system breaks down essay grading rubrics into disjoint components and grades each component separately, providing more detailed and transparent assessment.

## Architecture

### 1. Rubric Parser (`src/rubric_parser/parser.py`)
- Extracts disjoint grading categories from rubrics
- Uses Claude Opus to identify categories like:
  - Position & Insight
  - Use of Sources
  - Organization & Coherence
  - Language & Style
  - Grammar & Mechanics
- Each category includes score descriptors for values 1-6

### 2. Prompt Generator (`src/prompt_generator/generator.py`)
- Creates specialized grading prompts for each component
- Uses Claude Opus to generate component-specific evaluation prompts
- Ensures focused assessment on single aspects

### 3. Component Grader (`src/ai_agent/component_grader.py`)
- Grades essays on each component separately
- Supports different models for different components
- Calculates final score as average of component scores

### 4. AI Client Factory (`src/ai_agent/ai_client_factory.py`)
- Supports multiple AI providers:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude Opus, Sonnet, Haiku)
  - Google Gemini (Flash, Pro)
- Easy model switching via configuration

## Usage

### Basic Usage

```bash
python scripts/grade_essays_component.py
```

### Configuration

Edit the configuration section in `scripts/grade_essays_component.py`:

```python
# Model configuration
rubric_parser_model = "anthropic:opus"  # For parsing rubric
prompt_gen_model = "anthropic:opus"     # For generating prompts

# Grading models per component
grading_models = {
    "Position & Insight": "anthropic:sonnet",
    "Use of Sources": "openai:gpt-4",
    "Organization & Coherence": "gemini:flash",
    "Language & Style": "anthropic:sonnet",
    "Grammar & Mechanics": "openai:gpt-4-mini"
}
```

### Model Specifications

You can specify models in several ways:
- Provider only: `"openai"`, `"anthropic"`, `"gemini"`
- Model alias: `"gpt4"`, `"claude-opus"`, `"gemini-pro"`
- Full specification: `"openai:gpt-4"`, `"anthropic:claude-3-opus-20240229"`

### Environment Variables

Set API keys in `.env`:
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
SHEETS_CREDENTIALS_BASE64=your_sheets_credentials
```

## Output Format

The system outputs to Google Sheets with:
- **Run ID**: Timestamp of the grading run
- **Essay ID**: Unique identifier
- **AI Score**: Overall score (average of components)
- **Actual Score**: Ground truth (if available)
- **Essay Text**: Full essay content
- **AI Reasoning**: Detailed component breakdowns
- **Model**: Models used for grading
- **QWK**: Quadratic Weighted Kappa (if actual scores available)
- **Component Scores**: Individual scores per category

### Component Score Format
```
Position & Insight: 5 | Use of Sources: 4 | Organization & Coherence: 5 | ...
```

## Benefits

1. **Transparency**: Clear breakdown of scores by component
2. **Flexibility**: Different models for different aspects
3. **Modularity**: Easy to add/modify components
4. **Detailed Feedback**: Component-specific reasoning
5. **Iterative Improvement**: Easy to test different model combinations

## Workflow

1. **Rubric Analysis**: Claude Opus analyzes rubric to extract categories
2. **Prompt Generation**: Claude Opus creates component-specific prompts
3. **Component Grading**: Each component graded separately (customizable models)
4. **Score Aggregation**: Final score = average of component scores
5. **Results Output**: Detailed results written to Google Sheets

## Default Categories

If AI parsing is skipped (`use_default_categories=True`), the system uses predefined categories based on common rubric structures:

- **Position & Insight** (Critical thinking and argumentation)
- **Use of Sources** (Evidence and citations)
- **Organization & Coherence** (Structure and flow)
- **Language & Style** (Vocabulary and sentence variety)
- **Grammar & Mechanics** (Technical writing quality)

## Performance Considerations

- **Concurrent Processing**: Configurable worker threads (`max_workers`)
- **API Rate Limits**: Consider provider limits when setting concurrency
- **Cost**: Component grading uses more API calls (n essays × m components)
- **Quality vs Speed**: Trade-off between detailed assessment and processing time

## Comparison with Holistic Grading

| Aspect | Holistic | Component-Based |
|--------|----------|-----------------|
| API Calls | 1 per essay | m per essay (m = components) |
| Transparency | Single score | Detailed breakdown |
| Flexibility | One model | Model per component |
| Feedback | General | Component-specific |
| Cost | Lower | Higher |
| Quality | Good | Potentially better |