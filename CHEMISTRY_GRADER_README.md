# Chemistry Criteria Grader

A comparison-based grading system for evaluating chemistry student investigation reports on individual criteria using pairwise comparisons.

## Overview

This grader evaluates chemistry reports based on 12 individual criteria from the QCAA Chemistry 2019 rubric. For each criterion, it performs pairwise comparisons between test reports and a set of sample reports with known scores, then uses scoring strategies to predict grades.

## Features

- **Individual Criteria Grading**: Grade reports on each of the 12 criteria separately
- **Pairwise Comparisons**: Uses GPT-5-mini (or other models) to compare reports
- **Multiple Output Formats**: Export results to Google Sheets or CSV files
- **Detailed Statistics**: Includes QWK scores, comparison matrices, and confidence levels
- **Flexible Scoring Strategies**: Supports 'original' and 'optimized' scoring methods

## Usage

### Basic Command

```bash
# Activate virtual environment
source venv/bin/activate

# Grade all criteria with default settings
python scripts/chemistry_criteria_grader.py

# Grade specific criteria
python scripts/chemistry_criteria_grader.py --criteria 1,2,3

# Grade criteria range
python scripts/chemistry_criteria_grader.py --criteria 1-6

# Limit number of test reports
python scripts/chemistry_criteria_grader.py --limit 10
```

### Output Options

#### Google Sheets Export
```bash
# Export to Google Sheets (requires credentials)
python scripts/chemistry_criteria_grader.py --spreadsheet-id YOUR_SHEET_ID

# Or set environment variable
export GOOGLE_SHEETS_ID=YOUR_SHEET_ID
python scripts/chemistry_criteria_grader.py
```

#### CSV Export
```bash
# Export to CSV files
python scripts/chemistry_criteria_grader.py --output-format csv

# Specify output directory
python scripts/chemistry_criteria_grader.py --output-format csv --output-dir ./results
```

### Advanced Options

```bash
# Use different AI model
python scripts/chemistry_criteria_grader.py --model openai:gpt-4

# Use different scoring strategy
python scripts/chemistry_criteria_grader.py --strategy optimized

# Enable verbose logging
python scripts/chemistry_criteria_grader.py --verbose

# Skip Google Sheets even if credentials are available
python scripts/chemistry_criteria_grader.py --no-sheets
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--criteria` | Criteria to grade (e.g., "1,2,3" or "1-5" or "1-12") | All criteria (1-12) |
| `--limit` | Number of test reports to grade | All available |
| `--model` | AI model for comparisons | openai:gpt-5-mini |
| `--strategy` | Scoring strategy ('original' or 'optimized') | original |
| `--spreadsheet-id` | Google Sheets ID for output | From env var |
| `--no-sheets` | Skip Google Sheets output | False |
| `--output-format` | Output format ('sheets' or 'csv') | sheets |
| `--output-dir` | Directory for file outputs | ./chemistry_output |
| `--verbose` | Enable verbose logging | False |

## Data Structure

### Input Files

The grader expects the following files in the `submissions/` directory:

1. **Report Files**: `submissions/assignments/S{ID}.txt`
   - Text files containing student chemistry investigation reports
   
2. **Grades File**: `submissions/grades/chemistry_grades.csv`
   - Contains actual grades for each student on each criterion
   
3. **Criteria Rubric**: `submissions/criterion/QCAA_Chemistry_2019_Criteria_breakdown.csv`
   - Defines the rubric descriptors for each criterion

### Comparison Set

The grader uses the first 6 reports as the comparison set (samples with known scores) and grades the remaining reports as test samples.

## Output Format

### Google Sheets Output

Each criterion gets its own worksheet with:
- Summary statistics (QWK score, number graded)
- Detailed results for each test report
- Comparison matrix showing wins/losses against samples
- Color-coded cells for easy visualization
- Win rates and confidence scores

### CSV Output

Creates one CSV file per criterion containing:
- Student ID
- Actual score and score band
- Predicted score
- Strategy used
- Number of comparisons

## Criteria Descriptions

The grader evaluates these 12 criteria:

1. **Criterion 1**: Rationale for the experiment
2. **Criterion 2**: Modifications to methodology
3. **Criterion 3**: Research question
4. **Criterion 4**: Methodology for data collection
5. **Criterion 5**: Risk and ethical management
6. **Criterion 6**: Application of algorithms and data processing
7. **Criterion 7**: Identification of trends and patterns
8. **Criterion 8**: Uncertainty and limitations
9. **Criterion 9**: Investigation effectiveness
10. **Criterion 10**: Interpretation and conclusions
11. **Criterion 11**: Reliability and validity
12. **Criterion 12**: Improvements and extensions

## Scoring System

Each criterion uses a 4-level rubric:
- **5-6 points**: Highest level of achievement
- **3-4 points**: Satisfactory achievement
- **1-2 points**: Limited achievement
- **0 points**: Does not satisfy descriptors

## Implementation Details

The system consists of:

1. **Chemistry Data Loader**: Loads reports and rubrics
2. **Chemistry Comparison Engine**: Performs pairwise comparisons using AI
3. **Chemistry Criteria Grader**: Orchestrates grading process
4. **Chemistry Sheets Integration**: Formats and exports results
5. **Chemistry CLI**: Command-line interface

## Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Grade criteria 1-3 with first 10 test reports
python scripts/chemistry_criteria_grader.py --criteria 1-3 --limit 10

# 3. Export to Google Sheets
python scripts/chemistry_criteria_grader.py --criteria 1-3 --spreadsheet-id YOUR_ID

# 4. Or save as CSV files
python scripts/chemistry_criteria_grader.py --criteria 1-3 --output-format csv
```

## Notes

- The AI model defaults to `gpt-5-mini`
- Processing time is approximately 3-5 seconds per report per criterion
- Google Sheets credentials can be provided via environment variable `SHEETS_CREDENTIALS_BASE64`
- The grader maintains modularity with the existing essay grading system