"""Essay grading prompt."""

ESSAY_GRADING_PROMPT = """Please grade the following essay using the provided rubric. 

RUBRIC:
{text_rubric}

ESSAY TO GRADE:
{essay_text}

INSTRUCTIONS:
1. Carefully read the essay and evaluate it against each criterion in the rubric
2. Use the 1-6 scale as specified in the rubric (1 = lowest quality, 6 = highest quality)
3. Consider the weight of each criterion when determining the overall score
4. Provide detailed reasoning for your grade
5. Give a final score between 1 and 6 based on the rubric criteria

Please structure your response as follows:

REASONING:
[Provide detailed analysis of the essay against each criterion. Explain strengths and areas for improvement. Reference specific aspects from the rubric.]

FINAL SCORE: [Your numerical score between 1 and 6 based on the rubric criteria]

Be specific in your feedback and justify your scoring decisions based on the rubric criteria.""" 