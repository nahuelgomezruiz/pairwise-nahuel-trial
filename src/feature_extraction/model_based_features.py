"""
Model-based feature extraction using GPT-5-nano.
Implements section O features from the essay grading feature list.
"""

import json
import logging
import statistics
from typing import Dict, List, Optional, Any
import threading
import time
import openai
from openai import OpenAI
import time
import re


class ModelBasedFeatureExtractor:
    """Extracts model-based features using GPT-5-nano."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-5-nano",
        max_concurrency: int = 256,
        rpm_limit: Optional[int] = 30000,
        tpm_limit: Optional[int] = 180_000_000,
    ):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        # Simple client-side rate limiting
        self._semaphore = threading.Semaphore(max(1, max_concurrency))
        self._rpm_limit = rpm_limit or 30000
        self._tpm_limit = tpm_limit or 180_000_000
        self._last_minute = int(time.time() // 60)
        self._requests_this_min = 0
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to use environment variable
            try:
                self.client = OpenAI()
            except Exception as e:
                self.logger.warning(f"Could not initialize OpenAI client: {e}")
                self.client = None
    
    def extract_all_features(self, essay_text: str, 
                           prompt_text: Optional[str] = None,
                           source_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract all model-based features from an essay."""
        
        if not self.client:
            self.logger.warning("OpenAI client not available, returning zero features")
            return self._get_zero_features()
        
        features = {}
        
        try:
            # 88) Prompt adherence (topic and task fit)
            if prompt_text:
                prompt_features = self._extract_prompt_adherence(essay_text, prompt_text)
                features.update(prompt_features)
            
            # 89) Evidence grounding against source texts
            if source_texts:
                evidence_features = self._extract_evidence_grounding(essay_text, source_texts)
                features.update(evidence_features)
            
            # 90) Thesis (central claim) detection
            thesis_features = self._extract_thesis_detection(essay_text)
            features.update(thesis_features)
            
            # 91) Evidence-explanation linkage
            linkage_features = self._extract_evidence_explanation_linkage(essay_text)
            features.update(linkage_features)
            
        except Exception as e:
            self.logger.error(f"Error in model-based feature extraction: {e}")
            # Return partial features with zeros for failed extractions
            features.update(self._get_zero_features())
        
        return features
    
    def _call_gpt5_nano(self, messages: List[Dict[str, str]],
                         max_tokens: int = 500,
                         temperature: Optional[float] = None) -> Optional[str]:
        """Make a call to GPT-5-nano with error handling and retries."""
        if not self.client:
            return None
        # Basic per-minute request limiter aligned to RPM
        now_min = int(time.time() // 60)
        if now_min != self._last_minute:
            self._last_minute = now_min
            self._requests_this_min = 0
        if self._requests_this_min >= self._rpm_limit:
            sleep_for = (self._last_minute + 1) * 60 - time.time()
            if sleep_for > 0:
                time.sleep(min(sleep_for, 1.0))
        
        # Concurrency guard
        self._semaphore.acquire()
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                # Some nano models only accept default temperature; omit if not explicitly provided
                if temperature is not None:
                    kwargs['temperature'] = temperature
                response = self.client.chat.completions.create(**kwargs)
                self._requests_this_min += 1
                self._semaphore.release()
                return response.choices[0].message.content
                
            except openai.RateLimitError:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Rate limit hit, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error("Rate limit exceeded, max retries reached")
                    self._semaphore.release()
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error calling GPT-5-nano (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self._semaphore.release()
                    return None
        
        self._semaphore.release()
        return None
    
    def _extract_prompt_adherence(self, essay_text: str, prompt_text: str) -> Dict[str, float]:
        """
        Extract prompt adherence features.
        
        Outputs:
        - Average similarity between essay sentences and the prompt
        - Share of sentences that align with the prompt (coverage)
        - Off-topic rate (sentences below a similarity floor)
        - Task-match flag (argument, explanation, comparison, as requested)
        """
        
        # Split essay into sentences
        sentences = self._split_into_sentences(essay_text)
        
        if not sentences:
            return {
                'prompt_similarity_avg': 0.0,
                'prompt_coverage_rate': 0.0,
                'off_topic_rate': 0.0,
                'task_match_score': 0.0
            }
        
        # Analyze sentence-level similarity to prompt
        similarity_prompt = f"""
        Analyze how well each sentence in the following essay aligns with the given prompt.
        
        PROMPT: {prompt_text}
        
        ESSAY: {essay_text}
        
        For each sentence, rate its relevance to the prompt on a scale of 0-10 (0 = completely off-topic, 10 = perfectly aligned).
        
        Respond with a JSON object containing:
        - "sentence_scores": list of scores (0-10) for each sentence
        - "overall_task_match": score (0-10) for how well the essay addresses the prompt's task
        - "task_type_detected": the type of task you detect (e.g., "argument", "explanation", "comparison", "analysis")
        
        Be concise and focus on relevance scoring.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert essay evaluator. Analyze prompt adherence objectively and provide numerical scores."},
            {"role": "user", "content": similarity_prompt}
        ]
        
        response = self._call_gpt5_nano(messages, max_tokens=800)
        
        if not response:
            return self._get_prompt_adherence_defaults()
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                sentence_scores = result.get('sentence_scores', [])
                task_match = result.get('overall_task_match', 0)
                
                if sentence_scores:
                    # Convert 0-10 scores to 0-1 range
                    normalized_scores = [score / 10.0 for score in sentence_scores]
                    
                    avg_similarity = statistics.mean(normalized_scores)
                    coverage_rate = sum(1 for score in normalized_scores if score >= 0.5) / len(normalized_scores)
                    off_topic_rate = sum(1 for score in normalized_scores if score < 0.3) / len(normalized_scores)
                    
                    return {
                        'prompt_similarity_avg': avg_similarity,
                        'prompt_coverage_rate': coverage_rate,
                        'off_topic_rate': off_topic_rate,
                        'task_match_score': task_match / 10.0
                    }
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not parse prompt adherence response: {e}")
        
        return self._get_prompt_adherence_defaults()
    
    def _extract_evidence_grounding(self, essay_text: str, source_texts: List[str]) -> Dict[str, float]:
        """
        Extract evidence grounding features.
        
        Outputs:
        - Supported-claim rate (claims linked to at least one source passage)
        - Contradiction rate
        - Misattribution count (wrong source credited)
        - Paraphrase-versus-quote support rate
        """
        
        sources_summary = "\n\n".join([f"SOURCE {i+1}: {text[:500]}..." 
                                     for i, text in enumerate(source_texts)])
        
        grounding_prompt = f"""
        Analyze how well the following essay uses evidence from the provided sources.
        
        SOURCES:
        {sources_summary}
        
        ESSAY: {essay_text}
        
        Evaluate and respond with a JSON object containing:
        - "supported_claims": number of claims that are properly supported by sources
        - "total_claims": total number of claims made in the essay
        - "contradictions": number of claims that contradict the sources
        - "misattributions": number of times evidence is attributed to the wrong source
        - "quotes_vs_paraphrases": ratio of direct quotes to paraphrased evidence (0-1, where 1 = all quotes)
        
        Focus on factual accuracy and proper attribution.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at evaluating source usage and evidence grounding in academic writing."},
            {"role": "user", "content": grounding_prompt}
        ]
        
        response = self._call_gpt5_nano(messages, max_tokens=600)
        
        if not response:
            return self._get_evidence_grounding_defaults()
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                supported = result.get('supported_claims', 0)
                total = result.get('total_claims', 1)  # Avoid division by zero
                contradictions = result.get('contradictions', 0)
                misattributions = result.get('misattributions', 0)
                quote_ratio = result.get('quotes_vs_paraphrases', 0.5)
                
                return {
                    'supported_claim_rate': supported / max(total, 1),
                    'contradiction_rate': contradictions / max(total, 1),
                    'misattribution_count': float(misattributions),
                    'quote_vs_paraphrase_ratio': float(quote_ratio)
                }
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not parse evidence grounding response: {e}")
        
        return self._get_evidence_grounding_defaults()
    
    def _extract_thesis_detection(self, essay_text: str) -> Dict[str, float]:
        """
        Extract thesis detection features.
        
        Outputs:
        - Thesis present (0 or 1)
        - Thesis position (from start to end of the essay)
        - Thesis specificity score (generic versus specific wording)
        """
        
        thesis_prompt = f"""
        Analyze the following essay to identify its thesis statement (main argument/claim).
        
        ESSAY: {essay_text}
        
        Respond with a JSON object containing:
        - "thesis_present": 1 if a clear thesis is present, 0 if not
        - "thesis_position": position in the essay as a percentage (0-100, where 0 = very beginning, 100 = very end)
        - "thesis_specificity": how specific the thesis is (0-10, where 0 = very generic, 10 = very specific)
        - "thesis_text": the actual thesis statement if found (or "None" if not found)
        
        A thesis should be a clear, arguable claim that the essay supports.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at identifying thesis statements in academic essays."},
            {"role": "user", "content": thesis_prompt}
        ]
        
        response = self._call_gpt5_nano(messages, max_tokens=400)
        
        if not response:
            return self._get_thesis_detection_defaults()
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                thesis_present = result.get('thesis_present', 0)
                thesis_position = result.get('thesis_position', 0)
                thesis_specificity = result.get('thesis_specificity', 0)
                
                return {
                    'thesis_present': float(thesis_present),
                    'thesis_position_percent': float(thesis_position) / 100.0,
                    'thesis_specificity_score': float(thesis_specificity) / 10.0
                }
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not parse thesis detection response: {e}")
        
        return self._get_thesis_detection_defaults()
    
    def _extract_evidence_explanation_linkage(self, essay_text: str) -> Dict[str, float]:
        """
        Extract evidence-explanation linkage features.
        
        Outputs:
        - Explained-evidence rate (evidence followed by explanation that ties it to a claim)
        - Orphan-quote rate (evidence with no follow-up explanation)
        - Counterargument-and-refutation present (0 or 1)
        """
        
        linkage_prompt = f"""
        Analyze how well the following essay connects evidence to explanations and arguments.
        
        ESSAY: {essay_text}
        
        Evaluate and respond with a JSON object containing:
        - "evidence_pieces": total number of evidence pieces (quotes, citations, examples)
        - "explained_evidence": number of evidence pieces followed by clear explanation
        - "orphan_quotes": number of quotes/evidence with no explanation
        - "counterargument_present": 1 if counterarguments are addressed, 0 if not
        - "refutation_present": 1 if counterarguments are refuted, 0 if not
        
        Look for patterns like: Evidence → Explanation → Connection to main argument.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing argumentative structure and evidence usage in essays."},
            {"role": "user", "content": linkage_prompt}
        ]
        
        response = self._call_gpt5_nano(messages, max_tokens=400)
        
        if not response:
            return self._get_linkage_defaults()
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                evidence_pieces = result.get('evidence_pieces', 1)  # Avoid division by zero
                explained_evidence = result.get('explained_evidence', 0)
                orphan_quotes = result.get('orphan_quotes', 0)
                counterarg_present = result.get('counterargument_present', 0)
                refutation_present = result.get('refutation_present', 0)
                
                return {
                    'explained_evidence_rate': explained_evidence / max(evidence_pieces, 1),
                    'orphan_quote_rate': orphan_quotes / max(evidence_pieces, 1),
                    'counterargument_refutation_present': float(counterarg_present and refutation_present)
                }
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Could not parse linkage response: {e}")
        
        return self._get_linkage_defaults()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using a simple regex compatible with rule-based splitter."""
        sentences = re.split(r'[.!?]+\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_zero_features(self) -> Dict[str, float]:
        """Return dictionary with all model-based features set to zero."""
        return {
            # Prompt adherence
            'prompt_similarity_avg': 0.0,
            'prompt_coverage_rate': 0.0,
            'off_topic_rate': 0.0,
            'task_match_score': 0.0,
            
            # Evidence grounding
            'supported_claim_rate': 0.0,
            'contradiction_rate': 0.0,
            'misattribution_count': 0.0,
            'quote_vs_paraphrase_ratio': 0.0,
            
            # Thesis detection
            'thesis_present': 0.0,
            'thesis_position_percent': 0.0,
            'thesis_specificity_score': 0.0,
            
            # Evidence-explanation linkage
            'explained_evidence_rate': 0.0,
            'orphan_quote_rate': 0.0,
            'counterargument_refutation_present': 0.0
        }
    
    def _get_prompt_adherence_defaults(self) -> Dict[str, float]:
        """Default values for prompt adherence features (neutral zeros) plus availability flags elsewhere."""
        return {
            'prompt_similarity_avg': 0.0,
            'prompt_coverage_rate': 0.0,
            'off_topic_rate': 0.0,
            'task_match_score': 0.0
        }
    
    def _get_evidence_grounding_defaults(self) -> Dict[str, float]:
        """Default values for evidence grounding features (neutral zeros)."""
        return {
            'supported_claim_rate': 0.0,
            'contradiction_rate': 0.0,
            'misattribution_count': 0.0,
            'quote_vs_paraphrase_ratio': 0.0
        }
    
    def _get_thesis_detection_defaults(self) -> Dict[str, float]:
        """Default values for thesis detection features (neutral zeros)."""
        return {
            'thesis_present': 0.0,
            'thesis_position_percent': 0.0,
            'thesis_specificity_score': 0.0
        }
    
    def _get_linkage_defaults(self) -> Dict[str, float]:
        """Default values for evidence-explanation linkage features (neutral zeros)."""
        return {
            'explained_evidence_rate': 0.0,
            'orphan_quote_rate': 0.0,
            'counterargument_refutation_present': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of all model-based feature names."""
        return list(self._get_zero_features().keys())
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all model-based features."""
        return {
            'prompt_similarity_avg': 'Average similarity between essay sentences and the prompt',
            'prompt_coverage_rate': 'Share of sentences that align with the prompt',
            'off_topic_rate': 'Rate of sentences below similarity threshold',
            'task_match_score': 'How well essay addresses the prompt task',
            
            'supported_claim_rate': 'Rate of claims linked to source passages',
            'contradiction_rate': 'Rate of claims that contradict sources',
            'misattribution_count': 'Number of wrong source attributions',
            'quote_vs_paraphrase_ratio': 'Ratio of quotes to paraphrased evidence',
            
            'thesis_present': 'Whether a clear thesis is present (0 or 1)',
            'thesis_position_percent': 'Position of thesis in essay (0-1)',
            'thesis_specificity_score': 'How specific the thesis is (0-1)',
            
            'explained_evidence_rate': 'Rate of evidence followed by explanation',
            'orphan_quote_rate': 'Rate of evidence with no explanation',
            'counterargument_refutation_present': 'Whether counterarguments are addressed and refuted'
        }