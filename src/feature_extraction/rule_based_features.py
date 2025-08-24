"""
Rule-based feature extraction for essay grading.
Implements features A-N from the essay grading feature list.
"""

import re
import unicodedata
import string
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import cmudict
import syllapy

from .resource_manager import ResourceManager


class RuleBasedFeatureExtractor:
    """Extracts rule-based features from essays."""
    
    def __init__(self, resource_manager: ResourceManager, 
                 normalize_per_100_words: bool = True,
                 include_raw_counts: bool = True):
        self.resource_manager = resource_manager
        self.normalize_per_100_words = normalize_per_100_words
        self.include_raw_counts = include_raw_counts
        self.logger = logging.getLogger(__name__)
        self._syllable_cache: Dict[str, int] = {}
        
        # Download required NLTK data
        self._ensure_nltk_data()
        
        # Load pronunciation dictionary for syllable counting
        try:
            self.cmu_dict = cmudict.dict()
        except:
            self.cmu_dict = {}
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict')
    
    def extract_all_features(self, essay_text: str, 
                           prompt_text: Optional[str] = None,
                           source_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract all rule-based features from an essay."""
        
        # Normalize text
        normalized_text = self._normalize_text(essay_text)
        
        features = {}
        
        # A) Length and basic counts
        features.update(self._extract_length_features(normalized_text))
        
        # B) Punctuation and symbol use
        features.update(self._extract_punctuation_features(normalized_text))
        
        # C) Spelling and mechanics
        features.update(self._extract_spelling_features(normalized_text))
        
        # D) Sentence structure proxies
        features.update(self._extract_sentence_structure_features(normalized_text))
        
        # E) Organization and coherence
        features.update(self._extract_organization_features(normalized_text))
        
        # F) Alignment with assignment prompt (if available)
        if prompt_text:
            features.update(self._extract_prompt_alignment_features(normalized_text, prompt_text))
        
        # G) Use of sources and evidence
        features.update(self._extract_evidence_features(normalized_text, source_texts))
        
        # H) Reasoning and argument
        features.update(self._extract_reasoning_features(normalized_text))
        
        # I) Vocabulary and style
        features.update(self._extract_vocabulary_features(normalized_text))
        
        # J) Readability and fluency
        features.update(self._extract_readability_features(normalized_text))
        
        # K) Cohesion devices and discourse signals
        features.update(self._extract_cohesion_features(normalized_text))
        
        # L) Formatting and presentation
        features.update(self._extract_formatting_features(essay_text, prompt_text))
        
        # M) Integrity checks
        features.update(self._extract_integrity_features(normalized_text, source_texts))
        
        # N) Task-specific compliance
        if prompt_text:
            features.update(self._extract_compliance_features(normalized_text, prompt_text, source_texts))
        
        return features
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing."""
        # Unicode normalization and common punctuation normalization
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize('NFKC', text)
        # Map curly quotes and dashes
        replacements = {
            '“': '"', '”': '"', '„': '"', '‟': '"',
            '‘': "'", '’': "'", '‚': "'", '‛': "'",
            '–': '-', '—': '-', '−': '-',
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _get_sentences(self, text: str) -> List[str]:
        """Get list of sentences from text."""
        return sent_tokenize(text)
    
    def _get_words(self, text: str) -> List[str]:
        """Get list of words from text, preserving contractions and hyphenations."""
        text = text.lower()
        # Match alphabetic words possibly containing internal apostrophes or hyphens
        # e.g., don't, well-being, co-operate
        pattern = r"[a-z]+(?:['-][a-z]+)*"
        return re.findall(pattern, text)
    
    def _get_paragraphs(self, text: str) -> List[str]:
        """Get list of paragraphs from text."""
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        if word in self._syllable_cache:
            return self._syllable_cache[word]
        
        # Try CMU dictionary first
        if word in self.cmu_dict:
            count = len([phoneme for phoneme in self.cmu_dict[word][0] if phoneme[-1].isdigit()])
            self._syllable_cache[word] = count
            return count
        
        # Fallback to syllapy
        try:
            count = syllapy.count(word)
            self._syllable_cache[word] = count
            return count
        except:
            # Simple vowel counting as last resort
            vowels = 'aeiouy'
            count = 0
            prev_was_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
            count = max(1, count)
            self._syllable_cache[word] = count
            return count

    def _count_phrase_occurrences(self, text_lower: str, phrase: str) -> int:
        """Count whole-word occurrences of a (possibly multi-word) phrase."""
        # Escape regex meta in each token and join with flexible whitespace
        tokens = [re.escape(tok) for tok in phrase.split()]
        if not tokens:
            return 0
        pattern = r"\b" + r"\s+".join(tokens) + r"\b"
        return len(re.findall(pattern, text_lower))
    
    def _normalize_feature(self, count: int, word_count: int, per_n_words: int = 100) -> float:
        """Normalize count per N words."""
        if word_count == 0:
            return 0.0
        return (count / word_count) * per_n_words
    
    # A) Length and basic counts
    def _extract_length_features(self, text: str) -> Dict[str, float]:
        """Extract length and basic count features."""
        features = {}
        
        words = self._get_words(text)
        sentences = self._get_sentences(text)
        paragraphs = self._get_paragraphs(text)
        
        # 1) Total words
        total_words = len(words)
        features['total_words'] = float(total_words)
        
        # 2) Total characters
        features['total_chars_with_spaces'] = float(len(text))
        features['total_chars_no_spaces'] = float(len(text.replace(' ', '')))
        
        # 3) Sentence count
        sentence_count = len(sentences)
        features['sentence_count'] = float(sentence_count)
        
        # 4) Paragraph count
        paragraph_count = len(paragraphs)
        features['paragraph_count'] = float(paragraph_count)
        
        if sentence_count > 0:
            sentence_lengths = [len(self._get_words(sent)) for sent in sentences]
            
            # 5) Average sentence length
            features['avg_sentence_length'] = statistics.mean(sentence_lengths)
            
            # 6) Median sentence length
            features['median_sentence_length'] = statistics.median(sentence_lengths)
            
            # 7) Share of very short sentences (< 10 words)
            short_sentences = sum(1 for length in sentence_lengths if length < 10)
            features['short_sentence_rate'] = short_sentences / sentence_count
            
            # 8) Share of very long sentences (> 25 words)
            long_sentences = sum(1 for length in sentence_lengths if length > 25)
            features['long_sentence_rate'] = long_sentences / sentence_count
        else:
            features.update({
                'avg_sentence_length': 0.0,
                'median_sentence_length': 0.0,
                'short_sentence_rate': 0.0,
                'long_sentence_rate': 0.0
            })
        
        if total_words > 0:
            # 9) Unique word count
            unique_words = len(set(words))
            features['unique_word_count'] = float(unique_words)
            
            # 10) Lexical diversity
            features['lexical_diversity'] = unique_words / total_words
            
            # 11) Words used exactly once
            word_counts = Counter(words)
            once_words = sum(1 for count in word_counts.values() if count == 1)
            features['words_used_once_count'] = float(once_words)
            features['words_used_once_rate'] = once_words / total_words
            
            # 12) Long word rate (7+ characters)
            long_words = sum(1 for word in words if len(word) >= 7)
            features['long_word_count'] = float(long_words)
            features['long_word_rate'] = long_words / total_words
        else:
            features.update({
                'unique_word_count': 0.0,
                'lexical_diversity': 0.0,
                'words_used_once_count': 0.0,
                'words_used_once_rate': 0.0,
                'long_word_count': 0.0,
                'long_word_rate': 0.0
            })
        
        return features
    
    # B) Punctuation and symbol use
    def _extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation and symbol features."""
        features = {}
        words = self._get_words(text)
        word_count = len(words)
        
        # Count various punctuation marks
        # Compute counts carefully to avoid double counting quotes vs apostrophes
        quotation_marks_count = text.count('"')  # after normalization, curly quotes mapped to '"'
        # Apostrophes only when internal to a word (e.g., don't)
        apostrophes_internal = len(re.findall(r"(?i)(?<=\w)'(?=\w)", text))
        punctuation_counts = {
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'dash': text.count('-'),
            'parentheses': text.count('(') + text.count(')'),
            'brackets': text.count('[') + text.count(']'),
            'quotation_marks': quotation_marks_count,
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'ellipses': text.count('...'),
            'apostrophes': apostrophes_internal,
        }
        
        # Add counts and rates
        for punct_type, count in punctuation_counts.items():
            if self.include_raw_counts:
                features[f'{punct_type}_count'] = float(count)
            if self.normalize_per_100_words:
                features[f'{punct_type}_per_100_words'] = self._normalize_feature(count, word_count)
        
        # Unbalanced punctuation flags
        features['unmatched_quotes'] = float(abs(text.count('"') % 2))
        features['unmatched_parentheses'] = float(abs(text.count('(') - text.count(')')))
        features['unmatched_brackets'] = float(abs(text.count('[') - text.count(']')))
        
        # Double punctuation flags
        features['double_punctuation'] = float(
            len(re.findall(r'[!?]{2,}', text)) + 
            len(re.findall(r'\.{2,}', text))
        )
        
        # Double spaces after punctuation
        double_spaces = len(re.findall(r'[.!?]\s{2,}', text))
        features['double_spaces_count'] = float(double_spaces)
        if word_count > 0:
            features['double_spaces_rate'] = self._normalize_feature(double_spaces, word_count)
        else:
            features['double_spaces_rate'] = 0.0
        
        return features
    
    # C) Spelling and mechanics
    def _extract_spelling_features(self, text: str) -> Dict[str, float]:
        """Extract spelling and mechanics features."""
        features = {}
        words = self._get_words(text)
        word_count = len(words)
        
        if word_count == 0:
            return {
                'spelling_errors_count': 0.0,
                'spelling_errors_rate': 0.0,
                'unknown_words_count': 0.0,
                'unknown_words_rate': 0.0,
                'capitalization_errors': 0.0,
                'repeated_chars_count': 0.0,
                'repeated_words_count': 0.0
            }
        
        dictionary = self.resource_manager.get_english_dictionary()
        
        # Spelling errors (words not in dictionary)
        spelling_errors = sum(1 for word in words if word.lower() not in dictionary)
        features['spelling_errors_count'] = float(spelling_errors)
        features['spelling_errors_rate'] = self._normalize_feature(spelling_errors, word_count)
        
        # Unknown words (same as spelling errors for now)
        features['unknown_words_count'] = float(spelling_errors)
        features['unknown_words_rate'] = self._normalize_feature(spelling_errors, word_count)
        
        # Capitalization errors (simple heuristic)
        sentences = self._get_sentences(text)
        cap_errors = 0
        for sentence in sentences:
            # Find first alphabetic character
            m = re.search(r"[A-Za-z]", sentence)
            if m:
                if not sentence[m.start()].isupper():
                    cap_errors += 1
        features['capitalization_errors'] = float(cap_errors)
        
        # Repeated characters (e.g., "sooooo")
        repeated_chars = len(re.findall(r'(.)\1{2,}', text.lower()))
        features['repeated_chars_count'] = float(repeated_chars)
        
        # Repeated words (e.g., "very very")
        text_words = re.findall(r'\b\w+\b', text.lower())
        repeated_words = 0
        for i in range(len(text_words) - 1):
            if text_words[i] == text_words[i + 1]:
                repeated_words += 1
        features['repeated_words_count'] = float(repeated_words)
        
        return features
    
    # D) Sentence structure proxies
    def _extract_sentence_structure_features(self, text: str) -> Dict[str, float]:
        """Extract sentence structure proxy features."""
        features = {}
        words = self._get_words(text)
        word_count = len(words)
        sentences = self._get_sentences(text)
        sentence_count = len(sentences)
        
        # Get conjunction lists
        subordinators = self.resource_manager.get_subordinators()
        coordinators = self.resource_manager.get_coordinators()
        
        # Count subordinators and coordinators
        subordinator_count = sum(1 for word in words if word in subordinators)
        coordinator_count = sum(1 for word in words if word in coordinators)
        
        features['subordinator_count'] = float(subordinator_count)
        features['coordinator_count'] = float(coordinator_count)
        
        if word_count > 0:
            features['subordinator_rate'] = self._normalize_feature(subordinator_count, word_count)
            features['coordinator_rate'] = self._normalize_feature(coordinator_count, word_count)
        else:
            features['subordinator_rate'] = 0.0
            features['coordinator_rate'] = 0.0
        
        # Clause proxies: commas and semicolons per sentence
        if sentence_count > 0:
            total_commas = text.count(',')
            total_semicolons = text.count(';')
            features['commas_per_sentence'] = total_commas / sentence_count
            features['semicolons_per_sentence'] = total_semicolons / sentence_count
        else:
            features['commas_per_sentence'] = 0.0
            features['semicolons_per_sentence'] = 0.0
        
        # Multi-word subordinators/coordinators (phrase-aware)
        text_lower = text.lower()
        subordinator_phrases = [p for p in subordinators if ' ' in p]
        coordinator_phrases = [p for p in coordinators if ' ' in p]
        sub_phrase_count = sum(self._count_phrase_occurrences(text_lower, p) for p in subordinator_phrases)
        coord_phrase_count = sum(self._count_phrase_occurrences(text_lower, p) for p in coordinator_phrases)
        features['subordinator_phrase_count'] = float(sub_phrase_count)
        features['coordinator_phrase_count'] = float(coord_phrase_count)
        if word_count > 0:
            features['subordinator_phrase_rate'] = self._normalize_feature(sub_phrase_count, word_count)
            features['coordinator_phrase_rate'] = self._normalize_feature(coord_phrase_count, word_count)
        else:
            features['subordinator_phrase_rate'] = 0.0
            features['coordinator_phrase_rate'] = 0.0

        return features
    
    # E) Organization and coherence
    def _extract_organization_features(self, text: str) -> Dict[str, float]:
        """Extract organization and coherence features."""
        features = {}
        paragraphs = self._get_paragraphs(text)
        paragraph_count = len(paragraphs)
        
        if paragraph_count == 0:
            return {
                'avg_paragraph_length': 0.0,
                'paragraph_length_std': 0.0,
                'single_sentence_paragraphs': 0.0,
                'very_long_paragraphs': 0.0,
                'transition_phrase_count': 0.0,
                'transition_phrase_rate': 0.0,
                'transition_variety': 0.0,
                'local_cohesion_score': 0.0,
                'paragraph_continuity_score': 0.0,
                'repetition_score': 0.0
            }
        
        # Paragraph length statistics
        paragraph_lengths = [len(self._get_words(para)) for para in paragraphs]
        features['avg_paragraph_length'] = statistics.mean(paragraph_lengths)
        features['paragraph_length_std'] = statistics.stdev(paragraph_lengths) if len(paragraph_lengths) > 1 else 0.0
        
        # Single-sentence and very long paragraphs
        single_sent_paras = sum(1 for para in paragraphs if len(self._get_sentences(para)) == 1)
        very_long_paras = sum(1 for length in paragraph_lengths if length > 100)
        
        features['single_sentence_paragraphs'] = single_sent_paras / paragraph_count
        features['very_long_paragraphs'] = very_long_paras / paragraph_count
        
        # Transition phrases
        transition_phrases = self.resource_manager.get_transition_phrases()
        words = self._get_words(text)
        word_count = len(words)
        
        transition_count = 0
        used_transitions = set()
        
        # Check for transition phrases (both single words and multi-word phrases) with word boundaries
        text_lower = text.lower()
        for phrase in transition_phrases:
            phrase_count = self._count_phrase_occurrences(text_lower, phrase)
            transition_count += phrase_count
            if phrase_count > 0:
                used_transitions.add(phrase)
        
        features['transition_phrase_count'] = float(transition_count)
        features['transition_variety'] = float(len(used_transitions))
        
        if word_count > 0:
            features['transition_phrase_rate'] = self._normalize_feature(transition_count, word_count)
        else:
            features['transition_phrase_rate'] = 0.0
        
        # Local cohesion: content word overlap between neighboring sentences
        sentences = self._get_sentences(text)
        if len(sentences) > 1:
            stopwords = self.resource_manager.get_stopwords()
            cohesion_scores = []
            
            for i in range(len(sentences) - 1):
                words1 = set(word for word in self._get_words(sentences[i]) if word not in stopwords)
                words2 = set(word for word in self._get_words(sentences[i + 1]) if word not in stopwords)
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    total = len(words1.union(words2))
                    cohesion_scores.append(overlap / total if total > 0 else 0)
            
            features['local_cohesion_score'] = statistics.mean(cohesion_scores) if cohesion_scores else 0.0
        else:
            features['local_cohesion_score'] = 0.0
        
        # Paragraph-to-paragraph continuity
        if len(paragraphs) > 1:
            stopwords = self.resource_manager.get_stopwords()
            continuity_scores = []
            
            for i in range(len(paragraphs) - 1):
                words1 = set(word for word in self._get_words(paragraphs[i]) if word not in stopwords)
                words2 = set(word for word in self._get_words(paragraphs[i + 1]) if word not in stopwords)
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2))
                    total = len(words1.union(words2))
                    continuity_scores.append(overlap / total if total > 0 else 0)
            
            features['paragraph_continuity_score'] = statistics.mean(continuity_scores) if continuity_scores else 0.0
        else:
            features['paragraph_continuity_score'] = 0.0
        
        # Repetition score (repeated phrases)
        sentences = self._get_sentences(text)
        if len(sentences) > 1:
            sentence_similarities = []
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    words1 = set(self._get_words(sentences[i]))
                    words2 = set(self._get_words(sentences[j]))
                    if len(words1) > 0 and len(words2) > 0:
                        overlap = len(words1.intersection(words2))
                        total = len(words1.union(words2))
                        sentence_similarities.append(overlap / total if total > 0 else 0)
            
            features['repetition_score'] = statistics.mean(sentence_similarities) if sentence_similarities else 0.0
        else:
            features['repetition_score'] = 0.0
        
        return features
    
    # F) Alignment with assignment prompt
    def _extract_prompt_alignment_features(self, text: str, prompt_text: str) -> Dict[str, float]:
        """Extract prompt alignment features."""
        features = {}
        
        words = self._get_words(text)
        prompt_words = self._get_words(prompt_text)
        word_count = len(words)
        
        if word_count == 0 or len(prompt_words) == 0:
            return {
                'prompt_word_overlap_count': 0.0,
                'prompt_word_overlap_rate': 0.0,
                'word_count_in_range': 0.0
            }
        
        stopwords = self.resource_manager.get_stopwords()
        
        # Word overlap with prompt (excluding stopwords)
        essay_content_words = set(word for word in words if word not in stopwords)
        prompt_content_words = set(word for word in prompt_words if word not in stopwords)
        
        overlap_count = len(essay_content_words.intersection(prompt_content_words))
        features['prompt_word_overlap_count'] = float(overlap_count)
        features['prompt_word_overlap_rate'] = overlap_count / len(essay_content_words) if essay_content_words else 0.0
        
        # Word count in range (simple heuristic: 200-800 words is "good")
        target_min, target_max = 200, 800
        in_range = 1.0 if target_min <= word_count <= target_max else 0.0
        features['word_count_in_range'] = in_range
        
        return features
    
    # G) Use of sources and evidence
    def _extract_evidence_features(self, text: str, source_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract evidence and source usage features."""
        features = {}
        
        # Direct quotations
        # Support straight double quotes (curly mapped earlier)
        quotes = re.findall(r'"([^"]+)"', text)
        quote_count = len(quotes)
        
        words = self._get_words(text)
        word_count = len(words)
        
        features['quote_count'] = float(quote_count)
        
        if quote_count > 0:
            quote_lengths = [len(self._get_words(quote)) for quote in quotes]
            total_quoted_words = sum(quote_lengths)
            features['avg_quote_length'] = statistics.mean(quote_lengths)
            features['quoted_words_percentage'] = (total_quoted_words / word_count * 100) if word_count > 0 else 0.0
        else:
            features['avg_quote_length'] = 0.0
            features['quoted_words_percentage'] = 0.0
        
        # Attribution phrases
        reporting_verbs = self.resource_manager.get_reporting_verbs()
        attribution_count = 0
        text_lower = text.lower()
        
        for verb in reporting_verbs:
            attribution_count += self._count_phrase_occurrences(text_lower, verb)
        
        features['attribution_phrase_count'] = float(attribution_count)
        if word_count > 0:
            features['attribution_phrase_rate'] = self._normalize_feature(attribution_count, word_count)
        else:
            features['attribution_phrase_rate'] = 0.0
        
        # Quote proportion bands
        if word_count > 0:
            quote_percentage = features['quoted_words_percentage']
            if quote_percentage < 5:
                features['quote_band_very_low'] = 1.0
            elif quote_percentage < 15:
                features['quote_band_moderate'] = 1.0
            else:
                features['quote_band_very_high'] = 1.0
        
        # Initialize other features
        features.setdefault('quote_band_very_low', 0.0)
        features.setdefault('quote_band_moderate', 0.0)
        features.setdefault('quote_band_very_high', 0.0)
        
        # Source-specific features (if sources provided)
        if source_texts:
            # This would require more sophisticated matching
            features['source_mentions'] = 0.0  # Placeholder
            features['source_balance_score'] = 0.0  # Placeholder
            features['exact_quote_matches'] = 0.0  # Placeholder
            features['patchwriting_score'] = 0.0  # Placeholder
        
        return features
    
    # H) Reasoning and argument
    def _extract_reasoning_features(self, text: str) -> Dict[str, float]:
        """Extract reasoning and argument features."""
        features = {}
        
        text_lower = text.lower()
        words = self._get_words(text)
        word_count = len(words)
        
        # Counterargument signals
        counterarg_phrases = [
            'some may argue', 'critics contend', 'on the other hand',
            'opponents claim', 'however', 'nevertheless', 'although'
        ]
        
        counterarg_count = sum(text_lower.count(phrase) for phrase in counterarg_phrases)
        features['counterargument_signals'] = float(counterarg_count)
        
        # Refutation signals
        refutation_phrases = [
            'however', 'yet', 'nevertheless', 'this fails to account for',
            'but', 'despite this', 'on the contrary'
        ]
        
        refutation_count = sum(text_lower.count(phrase) for phrase in refutation_phrases)
        features['refutation_signals'] = float(refutation_count)
        
        # Reason and explanation markers
        reasoning_phrases = [
            'because', 'therefore', 'as a result', 'this shows that',
            'consequently', 'thus', 'hence', 'due to'
        ]
        
        reasoning_count = sum(text_lower.count(phrase) for phrase in reasoning_phrases)
        features['reasoning_markers'] = float(reasoning_count)
        
        # Normalize rates
        if word_count > 0:
            features['counterargument_rate'] = self._normalize_feature(counterarg_count, word_count)
            features['refutation_rate'] = self._normalize_feature(refutation_count, word_count)
            features['reasoning_rate'] = self._normalize_feature(reasoning_count, word_count)
        else:
            features['counterargument_rate'] = 0.0
            features['refutation_rate'] = 0.0
            features['reasoning_rate'] = 0.0
        
        return features
    
    # I) Vocabulary and style
    def _extract_vocabulary_features(self, text: str) -> Dict[str, float]:
        """Extract vocabulary and style features."""
        features = {}
        
        words = self._get_words(text)
        word_count = len(words)
        
        if word_count == 0:
            return {
                'rare_word_count': 0.0, 'rare_word_rate': 0.0,
                'academic_word_count': 0.0, 'academic_word_rate': 0.0,
                'vague_term_count': 0.0, 'vague_term_rate': 0.0,
                'hedging_count': 0.0, 'hedging_rate': 0.0,
                'certainty_count': 0.0, 'certainty_rate': 0.0,
                'cliche_count': 0.0, 'cliche_rate': 0.0,
                'contraction_count': 0.0, 'contraction_rate': 0.0,
                'uppercase_ratio': 0.0, 'digit_ratio': 0.0
            }
        
        # Get word lists
        frequencies = self.resource_manager.get_frequency_list()
        academic_words = self.resource_manager.get_academic_words()
        vague_terms = self.resource_manager.get_vague_terms()
        hedging_words = self.resource_manager.get_hedging_words()
        certainty_words = self.resource_manager.get_certainty_words()
        cliches = self.resource_manager.get_cliches()
        
        # Rare words (frequency < 1000 or not in frequency list)
        rare_word_count = sum(1 for word in words if frequencies.get(word, 0) < 1000)
        features['rare_word_count'] = float(rare_word_count)
        features['rare_word_rate'] = self._normalize_feature(rare_word_count, word_count)
        
        # Academic words
        academic_word_count = sum(1 for word in words if word in academic_words)
        features['academic_word_count'] = float(academic_word_count)
        features['academic_word_rate'] = self._normalize_feature(academic_word_count, word_count)
        
        # Vague terms
        text_lower = text.lower()
        vague_count = sum(self._count_phrase_occurrences(text_lower, term) for term in vague_terms)
        features['vague_term_count'] = float(vague_count)
        features['vague_term_rate'] = self._normalize_feature(vague_count, word_count)
        
        # Hedging words
        hedging_count = sum(self._count_phrase_occurrences(text_lower, word) for word in hedging_words)
        features['hedging_count'] = float(hedging_count)
        features['hedging_rate'] = self._normalize_feature(hedging_count, word_count)
        
        # Certainty words
        certainty_count = sum(self._count_phrase_occurrences(text_lower, word) for word in certainty_words)
        features['certainty_count'] = float(certainty_count)
        features['certainty_rate'] = self._normalize_feature(certainty_count, word_count)
        
        # Cliches
        cliche_count = sum(text_lower.count(cliche) for cliche in cliches)
        features['cliche_count'] = float(cliche_count)
        features['cliche_rate'] = self._normalize_feature(cliche_count, word_count)
        
        # Contractions
        contraction_pattern = r"\b\w+'\w+\b"
        contractions = re.findall(contraction_pattern, text)
        contraction_count = len(contractions)
        features['contraction_count'] = float(contraction_count)
        features['contraction_rate'] = self._normalize_feature(contraction_count, word_count)
        
        # Uppercase and digit ratios
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            uppercase_count = sum(1 for char in text if char.isupper())
            digit_count = sum(1 for char in text if char.isdigit())
            features['uppercase_ratio'] = uppercase_count / total_chars
            features['digit_ratio'] = digit_count / total_chars
        else:
            features['uppercase_ratio'] = 0.0
            features['digit_ratio'] = 0.0
        
        return features
    
    # J) Readability and fluency
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability and fluency features."""
        features = {}
        
        sentences = self._get_sentences(text)
        words = self._get_words(text)
        
        sentence_count = len(sentences)
        word_count = len(words)
        
        if sentence_count == 0 or word_count == 0:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'coleman_liau_index': 0.0,
                'gunning_fog_index': 0.0,
                'sentence_length_variance': 0.0
            }
        
        # Calculate syllable count
        total_syllables = sum(self._count_syllables(word) for word in words)
        
        # Average sentence length and syllables per word
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = total_syllables / word_count
        
        # Flesch Reading Ease
        flesch_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        features['flesch_reading_ease'] = flesch_ease
        
        # Flesch-Kincaid Grade Level
        flesch_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
        features['flesch_kincaid_grade'] = flesch_grade
        
        # Coleman-Liau Index
        # Letters only for Coleman–Liau
        letters_only = len(re.findall(r'[A-Za-z]', text))
        chars_per_100_words = (letters_only / word_count) * 100
        sentences_per_100_words = (sentence_count / word_count) * 100
        coleman_liau = (0.0588 * chars_per_100_words) - (0.296 * sentences_per_100_words) - 15.8
        features['coleman_liau_index'] = coleman_liau
        
        # Gunning Fog Index
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_ratio = complex_words / word_count
        gunning_fog = 0.4 * (avg_sentence_length + (100 * complex_word_ratio))
        features['gunning_fog_index'] = gunning_fog
        
        # Sentence length variance
        sentence_lengths = [len(self._get_words(sent)) for sent in sentences]
        if len(sentence_lengths) > 1:
            features['sentence_length_variance'] = statistics.variance(sentence_lengths)
        else:
            features['sentence_length_variance'] = 0.0
        
        return features
    
    # K) Cohesion devices and discourse signals
    def _extract_cohesion_features(self, text: str) -> Dict[str, float]:
        """Extract cohesion and discourse signal features."""
        features = {}
        
        text_lower = text.lower()
        words = self._get_words(text)
        word_count = len(words)
        
        # Example and definition signals
        example_signals = ['for example', 'for instance', 'such as', 'including', 'namely']
        definition_signals = ['is defined as', 'means', 'refers to', 'is known as']
        
        example_count = sum(text_lower.count(signal) for signal in example_signals)
        definition_count = sum(text_lower.count(signal) for signal in definition_signals)
        
        features['example_signals'] = float(example_count)
        features['definition_signals'] = float(definition_count)
        
        # Metadiscourse signals
        metadiscourse_signals = [
            'this essay shows', 'the following section', 'in this paragraph',
            'as mentioned above', 'as will be discussed', 'in summary'
        ]
        
        metadiscourse_count = sum(text_lower.count(signal) for signal in metadiscourse_signals)
        features['metadiscourse_signals'] = float(metadiscourse_count)
        
        # Normalize rates
        if word_count > 0:
            features['example_signals_rate'] = self._normalize_feature(example_count, word_count)
            features['definition_signals_rate'] = self._normalize_feature(definition_count, word_count)
            features['metadiscourse_rate'] = self._normalize_feature(metadiscourse_count, word_count)
        else:
            features['example_signals_rate'] = 0.0
            features['definition_signals_rate'] = 0.0
            features['metadiscourse_rate'] = 0.0
        
        return features
    
    # L) Formatting and presentation
    def _extract_formatting_features(self, text: str, prompt_text: Optional[str] = None) -> Dict[str, float]:
        """Extract formatting and presentation features."""
        features = {}
        
        # Title present (first line that's short and doesn't end with punctuation)
        lines = text.split('\n')
        has_title = 0.0
        if lines and len(lines[0].strip()) < 100 and not lines[0].strip().endswith(('.', '!', '?')):
            has_title = 1.0
        features['has_title'] = has_title
        
        # Paragraphing present
        paragraph_count = len(self._get_paragraphs(text))
        features['has_paragraphs'] = 1.0 if paragraph_count > 1 else 0.0
        
        # Citation indicators
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\b[A-Z][a-z]+ \(\d{4}\)',  # Author (2023)
            r'Works Cited',
            r'References',
            r'Bibliography'
        ]
        
        has_citations = 0.0
        for pattern in citation_patterns:
            if re.search(pattern, text):
                has_citations = 1.0
                break
        
        features['has_citations'] = has_citations
        
        return features
    
    # M) Integrity checks
    def _extract_integrity_features(self, text: str, source_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract integrity check features."""
        features = {}
        
        # Self-duplication within essay
        sentences = self._get_sentences(text)
        
        if len(sentences) > 1:
            duplicate_sentences = 0
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    # Simple similarity check
                    words1 = set(self._get_words(sentences[i]))
                    words2 = set(self._get_words(sentences[j]))
                    if len(words1) > 0 and len(words2) > 0:
                        similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        if similarity > 0.8:  # High similarity threshold
                            duplicate_sentences += 1
            
            features['duplicate_sentences'] = float(duplicate_sentences)
        else:
            features['duplicate_sentences'] = 0.0
        
        # Source overlap (if sources provided)
        if source_texts:
            # Placeholder for source overlap detection
            features['source_overlap_score'] = 0.0
        else:
            features['source_overlap_score'] = 0.0
        
        return features
    
    # N) Task-specific compliance
    def _extract_compliance_features(self, text: str, prompt_text: str, 
                                   source_texts: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract task-specific compliance features."""
        features = {}
        
        words = self._get_words(text)
        word_count = len(words)
        
        # Word count compliance (heuristic ranges)
        # This would need to be extracted from the actual prompt
        target_ranges = {
            'short': (200, 400),
            'medium': (400, 800),
            'long': (800, 1200)
        }
        
        # Default to medium range
        min_words, max_words = target_ranges['medium']
        in_range = 1.0 if min_words <= word_count <= max_words else 0.0
        features['word_count_compliance'] = in_range
        
        # Required sections (would need prompt parsing)
        # Placeholder implementation
        features['required_sections_present'] = 1.0  # Assume present for now
        
        # Minimum sources (if applicable)
        if source_texts:
            features['minimum_sources_met'] = 1.0  # Placeholder
        else:
            features['minimum_sources_met'] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all rule-based feature names."""
        # This would return all feature names that can be extracted
        # For now, return a sample based on the implemented features
        sample_text = "This is a sample essay with multiple sentences. It has paragraphs too."
        sample_features = self.extract_all_features(sample_text)
        return list(sample_features.keys())
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all rule-based features."""
        return {
            'total_words': 'Total number of words in the essay',
            'sentence_count': 'Total number of sentences',
            'paragraph_count': 'Total number of paragraphs',
            'avg_sentence_length': 'Average sentence length in words',
            'lexical_diversity': 'Ratio of unique words to total words',
            'comma_per_100_words': 'Number of commas per 100 words',
            'spelling_errors_rate': 'Rate of spelling errors per 100 words',
            'transition_phrase_rate': 'Rate of transition phrases per 100 words',
            'flesch_reading_ease': 'Flesch Reading Ease score',
            # Add more descriptions as needed
        }