# Essay Feature Extraction System

A comprehensive system for extracting essay grading features using both rule-based and AI model-based approaches.

## Overview

This system implements **91 distinct features** from the essay grading feature list, covering:

- **Rule-based features (A-N)**: Length, punctuation, spelling, sentence structure, organization, vocabulary, readability, and more
- **Model-based features (O)**: GPT-5-nano powered analysis for prompt adherence, evidence grounding, thesis detection, and argument structure

## Features Implemented

### Rule-Based Features (Sections A-N)

1. **Length and Basic Counts** (12 features)
   - Word count, character count, sentence/paragraph counts
   - Lexical diversity, unique words, sentence length statistics

2. **Punctuation and Symbol Use** (25 features)
   - Comma, semicolon, colon, dash, quotation mark rates
   - Unbalanced punctuation detection, double punctuation flags

3. **Spelling and Mechanics** (7 features)
   - Spelling error rates, unknown words, capitalization errors
   - Repeated character/word detection

4. **Sentence Structure Proxies** (6 features)
   - Subordinating/coordinating conjunction rates
   - Clause complexity indicators

5. **Organization and Coherence** (9 features)
   - Paragraph statistics, transition phrase usage
   - Local cohesion, paragraph continuity, repetition analysis

6. **Prompt Alignment** (3 features)
   - Word overlap with prompt, compliance indicators

7. **Evidence and Source Usage** (8 features)
   - Quotation analysis, attribution phrases, quote proportion bands

8. **Reasoning and Argument** (6 features)
   - Counterargument signals, refutation markers, reasoning indicators

9. **Vocabulary and Style** (12 features)
   - Rare words, academic vocabulary, vagueness, hedging
   - Certainty markers, clichés, contractions

10. **Readability and Fluency** (5 features)
    - Flesch Reading Ease, Flesch-Kincaid Grade Level
    - Coleman-Liau Index, Gunning Fog Index

11. **Cohesion and Discourse** (6 features)
    - Example/definition signals, metadiscourse markers

12. **Formatting and Presentation** (3 features)
    - Title presence, paragraphing, citation indicators

13. **Integrity Checks** (2 features)
    - Self-duplication detection, source overlap analysis

14. **Task Compliance** (3 features)
    - Word count compliance, required sections, source requirements

### Model-Based Features (Section O)

Uses **GPT-5-nano** for sophisticated analysis:

1. **Prompt Adherence** (4 features)
   - Sentence-level similarity to prompt
   - Coverage rate, off-topic detection, task match scoring

2. **Evidence Grounding** (4 features)
   - Supported claim rate, contradiction detection
   - Misattribution analysis, quote vs. paraphrase ratios

3. **Thesis Detection** (3 features)
   - Thesis presence, position, specificity scoring

4. **Evidence-Explanation Linkage** (3 features)
   - Explained evidence rate, orphan quote detection
   - Counterargument-refutation analysis

## Installation

1. **Clone and setup environment:**
```bash
cd nahuel-trial
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements_features.txt
```

3. **For model-based features, get OpenAI API key:**
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Get API key with GPT-5-nano access
   - Set environment variable: `export OPENAI_API_KEY=your_key_here`

## Usage

### Basic Feature Extraction

```bash
# Rule-based features only (no API key needed)
python scripts/extract_essay_features.py \
    --input data/essays.csv \
    --output features.csv \
    --no-model-features

# Full feature extraction with GPT-5-nano
python scripts/extract_essay_features.py \
    --input data/essays.csv \
    --output features.csv \
    --api-key YOUR_API_KEY
```

### With Prompt and Sources

```bash
python scripts/extract_essay_features.py \
    --input data/essays.csv \
    --output features.csv \
    --prompt assignment_prompt.txt \
    --sources source_texts.csv \
    --api-key YOUR_API_KEY
```

### Python API Usage

```python
from feature_extraction.feature_extractor import FeatureExtractor, FeatureExtractionConfig

# Configure extraction
config = FeatureExtractionConfig(
    openai_api_key="your_key_here",
    model_name="gpt-5-nano",
    include_rule_based=True,
    include_model_based=True
)

# Initialize extractor
extractor = FeatureExtractor(config)

# Extract features from single essay
features = extractor.extract_features(
    essay_text="Your essay text here...",
    prompt_text="Assignment prompt...",  # optional
    source_texts=["Source 1...", "Source 2..."]  # optional
)

# Extract from multiple essays
essays = [
    {'essay_text': 'Essay 1...', 'essay_id': 1},
    {'essay_text': 'Essay 2...', 'essay_id': 2}
]
features_df = extractor.extract_features_batch(essays)
```

## Complete Example: Training a Model

```python
# Run the complete example
python example_usage.py
```

This will:
1. Extract features from training data
2. Split into train/test sets (80%/20%)
3. Train a Random Forest regressor
4. Evaluate performance and show feature importance
5. Generate plots and save results

## Input Data Format

Your CSV file should have these columns:

- `essay_id`: Unique identifier for each essay
- `full_text`: The essay text content
- `score`: Target score (for training data)

Example:
```csv
essay_id,full_text,score
001,"This is the essay text...",4
002,"Another essay here...",3
```

## Output

The system generates:

1. **Features CSV**: All extracted features with essay IDs and scores
2. **Metadata file**: Summary of extraction process and feature list
3. **Feature importance**: Ranking of most predictive features
4. **Model performance**: R², MSE, MAE metrics
5. **Visualizations**: Actual vs. predicted plots, residual analysis

## Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Length & Structure | 12 | word_count, avg_sentence_length, lexical_diversity |
| Punctuation | 25 | comma_per_100_words, unmatched_quotes |
| Language Quality | 19 | spelling_errors_rate, academic_word_rate |
| Organization | 15 | transition_phrase_rate, local_cohesion_score |
| Evidence Use | 8 | quote_count, attribution_phrase_rate |
| Reasoning | 6 | counterargument_signals, reasoning_markers |
| Style & Voice | 12 | rare_word_rate, hedging_rate, certainty_rate |
| Readability | 5 | flesch_reading_ease, gunning_fog_index |
| AI-Powered | 14 | prompt_similarity_avg, thesis_present, supported_claim_rate |

## Performance Notes

- **Rule-based features**: Fast, no API costs
- **Model-based features**: Slower, requires OpenAI API credits
- **Batch processing**: Recommended for large datasets
- **Memory usage**: ~100MB per 1000 essays

## GPT-5-nano Integration

The system uses GPT-5-nano for advanced analysis:

- **Cost**: ~$0.05 per 1M input tokens, $0.40 per 1M output tokens
- **Speed**: Optimized for low latency
- **Reliability**: Built-in retry logic and error handling
- **Fallbacks**: Graceful degradation if API unavailable

## Extending the System

Add new features by:

1. **Rule-based**: Extend `RuleBasedFeatureExtractor`
2. **Model-based**: Extend `ModelBasedFeatureExtractor`
3. **Resources**: Add word lists to `ResourceManager`

## Troubleshooting

**Common Issues:**

1. **Import errors**: Ensure `venv` is activated and dependencies installed
2. **API errors**: Check OpenAI API key and rate limits
3. **Memory issues**: Reduce batch size or process in chunks
4. **Missing features**: Check input data format and column names

**Getting Help:**

- Check logs in `feature_extraction_YYYYMMDD_HHMMSS.log`
- Review metadata files for extraction summaries
- Use `--log-level DEBUG` for detailed output

## Citation

If you use this system in research, please cite:

```
Essay Feature Extraction System
Comprehensive rule-based and AI-powered feature extraction for automated essay scoring
2025
```