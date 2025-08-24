"""
Resource manager for loading word lists, dictionaries, and other resources
needed for feature extraction.
"""

import os
import json
import pickle
from typing import Set, List, Dict, Optional
import logging
from pathlib import Path


class ResourceManager:
    """Manages loading and caching of linguistic resources."""
    
    def __init__(self, resources_dir: str = "resources"):
        self.resources_dir = Path(resources_dir)
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        
        # Ensure resources directory exists
        self.resources_dir.mkdir(exist_ok=True)
        
        # Initialize built-in resources if they don't exist
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize built-in word lists and resources."""
        
        # Create basic word lists if they don't exist
        self._create_stopwords()
        self._create_transition_phrases()
        self._create_reporting_verbs()
        self._create_vague_terms()
        self._create_hedging_words()
        self._create_certainty_words()
        self._create_cliches()
        self._create_subordinators()
        self._create_coordinators()
        self._create_academic_words()
    
    def get_english_dictionary(self) -> Set[str]:
        """Get English dictionary word set."""
        if 'english_dictionary' not in self._cache:
            dict_path = self.resources_dir / "english_dictionary.txt"
            
            if dict_path.exists():
                with open(dict_path, 'r', encoding='utf-8') as f:
                    words = {line.strip().lower() for line in f if line.strip()}
            else:
                # Create a basic dictionary from common words
                words = self._create_basic_dictionary()
                self._save_word_list(words, dict_path)
            
            self._cache['english_dictionary'] = words
        
        return self._cache['english_dictionary']
    
    def get_stopwords(self) -> Set[str]:
        """Get stopword set."""
        return self._get_word_set('stopwords.txt', 'stopwords')
    
    def get_transition_phrases(self) -> List[str]:
        """Get transition phrase list."""
        return self._get_word_list('transition_phrases.txt', 'transition_phrases')
    
    def get_reporting_verbs(self) -> List[str]:
        """Get reporting verb list."""
        return self._get_word_list('reporting_verbs.txt', 'reporting_verbs')
    
    def get_vague_terms(self) -> List[str]:
        """Get vague terms list."""
        return self._get_word_list('vague_terms.txt', 'vague_terms')
    
    def get_hedging_words(self) -> List[str]:
        """Get hedging words list."""
        return self._get_word_list('hedging_words.txt', 'hedging_words')
    
    def get_certainty_words(self) -> List[str]:
        """Get certainty words list."""
        return self._get_word_list('certainty_words.txt', 'certainty_words')
    
    def get_cliches(self) -> List[str]:
        """Get cliche phrases list."""
        return self._get_word_list('cliches.txt', 'cliches')
    
    def get_subordinators(self) -> List[str]:
        """Get subordinating conjunctions."""
        return self._get_word_list('subordinators.txt', 'subordinators')
    
    def get_coordinators(self) -> List[str]:
        """Get coordinating conjunctions."""
        return self._get_word_list('coordinators.txt', 'coordinators')
    
    def get_academic_words(self) -> Set[str]:
        """Get academic word list."""
        return self._get_word_set('academic_words.txt', 'academic_words')
    
    def get_frequency_list(self) -> Dict[str, int]:
        """Get word frequency dictionary."""
        if 'frequency_list' not in self._cache:
            freq_path = self.resources_dir / "word_frequencies.json"
            
            if freq_path.exists():
                with open(freq_path, 'r', encoding='utf-8') as f:
                    frequencies = json.load(f)
            else:
                # Create basic frequency list
                frequencies = self._create_basic_frequencies()
                with open(freq_path, 'w', encoding='utf-8') as f:
                    json.dump(frequencies, f, indent=2)
            
            self._cache['frequency_list'] = frequencies
        
        return self._cache['frequency_list']
    
    def _get_word_set(self, filename: str, cache_key: str) -> Set[str]:
        """Get a word set from file with caching."""
        if cache_key not in self._cache:
            file_path = self.resources_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    words = {line.strip().lower() for line in f if line.strip()}
            else:
                words = set()
            self._cache[cache_key] = words
        
        return self._cache[cache_key]
    
    def _get_word_list(self, filename: str, cache_key: str) -> List[str]:
        """Get a word list from file with caching."""
        if cache_key not in self._cache:
            file_path = self.resources_dir / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    words = [line.strip().lower() for line in f if line.strip()]
            else:
                words = []
            self._cache[cache_key] = words
        
        return self._cache[cache_key]
    
    def _save_word_list(self, words: Set[str], file_path: Path):
        """Save word list to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for word in sorted(words):
                f.write(f"{word}\n")
    
    def _create_stopwords(self):
        """Create basic stopwords list."""
        stopwords_path = self.resources_dir / "stopwords.txt"
        if not stopwords_path.exists():
            stopwords = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'i', 'you', 'we', 'they', 'she', 'her',
                'his', 'him', 'this', 'these', 'those', 'there', 'their', 'them',
                'have', 'had', 'do', 'does', 'did', 'can', 'could', 'would', 'should'
            }
            self._save_word_list(stopwords, stopwords_path)
    
    def _create_transition_phrases(self):
        """Create transition phrases list."""
        phrases_path = self.resources_dir / "transition_phrases.txt"
        if not phrases_path.exists():
            phrases = [
                'however', 'therefore', 'furthermore', 'moreover', 'nevertheless',
                'consequently', 'in contrast', 'on the other hand', 'as a result',
                'in addition', 'for example', 'for instance', 'in conclusion',
                'to summarize', 'first', 'second', 'third', 'finally', 'meanwhile',
                'subsequently', 'similarly', 'likewise', 'conversely', 'nonetheless'
            ]
            with open(phrases_path, 'w', encoding='utf-8') as f:
                for phrase in phrases:
                    f.write(f"{phrase}\n")
    
    def _create_reporting_verbs(self):
        """Create reporting verbs list."""
        verbs_path = self.resources_dir / "reporting_verbs.txt"
        if not verbs_path.exists():
            verbs = [
                'states', 'argues', 'claims', 'suggests', 'reports', 'writes',
                'explains', 'describes', 'notes', 'observes', 'mentions',
                'indicates', 'proposes', 'asserts', 'maintains', 'contends',
                'according to', 'the author', 'the article', 'the passage'
            ]
            with open(verbs_path, 'w', encoding='utf-8') as f:
                for verb in verbs:
                    f.write(f"{verb}\n")
    
    def _create_vague_terms(self):
        """Create vague terms list."""
        terms_path = self.resources_dir / "vague_terms.txt"
        if not terms_path.exists():
            terms = [
                'things', 'stuff', 'a lot', 'very', 'really', 'quite', 'rather',
                'somewhat', 'kind of', 'sort of', 'basically', 'generally',
                'usually', 'often', 'sometimes', 'many', 'some', 'few'
            ]
            with open(terms_path, 'w', encoding='utf-8') as f:
                for term in terms:
                    f.write(f"{term}\n")
    
    def _create_hedging_words(self):
        """Create hedging words list."""
        words_path = self.resources_dir / "hedging_words.txt"
        if not words_path.exists():
            words = [
                'might', 'may', 'perhaps', 'possibly', 'probably', 'likely',
                'seems', 'appears', 'suggests', 'could', 'would', 'should',
                'tend to', 'inclined to', 'presumably', 'allegedly'
            ]
            with open(words_path, 'w', encoding='utf-8') as f:
                for word in words:
                    f.write(f"{word}\n")
    
    def _create_certainty_words(self):
        """Create certainty words list."""
        words_path = self.resources_dir / "certainty_words.txt"
        if not words_path.exists():
            words = [
                'obviously', 'clearly', 'undeniably', 'certainly', 'definitely',
                'absolutely', 'undoubtedly', 'surely', 'without question',
                'beyond doubt', 'unquestionably', 'indubitably'
            ]
            with open(words_path, 'w', encoding='utf-8') as f:
                for word in words:
                    f.write(f"{word}\n")
    
    def _create_cliches(self):
        """Create cliches list."""
        cliches_path = self.resources_dir / "cliches.txt"
        if not cliches_path.exists():
            cliches = [
                'in today\'s society', 'since the dawn of time', 'throughout history',
                'in this day and age', 'as we all know', 'it goes without saying',
                'needless to say', 'last but not least', 'first and foremost',
                'at the end of the day', 'when all is said and done'
            ]
            with open(cliches_path, 'w', encoding='utf-8') as f:
                for cliche in cliches:
                    f.write(f"{cliche}\n")
    
    def _create_subordinators(self):
        """Create subordinating conjunctions list."""
        sub_path = self.resources_dir / "subordinators.txt"
        if not sub_path.exists():
            subordinators = [
                'because', 'although', 'since', 'while', 'if', 'unless',
                'whereas', 'though', 'when', 'where', 'after', 'before',
                'until', 'as', 'even though', 'provided that', 'in order that'
            ]
            with open(sub_path, 'w', encoding='utf-8') as f:
                for sub in subordinators:
                    f.write(f"{sub}\n")
    
    def _create_coordinators(self):
        """Create coordinating conjunctions list."""
        coord_path = self.resources_dir / "coordinators.txt"
        if not coord_path.exists():
            coordinators = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so']
            with open(coord_path, 'w', encoding='utf-8') as f:
                for coord in coordinators:
                    f.write(f"{coord}\n")
    
    def _create_academic_words(self):
        """Create academic word list."""
        academic_path = self.resources_dir / "academic_words.txt"
        if not academic_path.exists():
            # Sample academic words from Coxhead's Academic Word List
            academic_words = {
                'analyze', 'approach', 'area', 'assess', 'assume', 'authority',
                'available', 'benefit', 'concept', 'consistent', 'constitute',
                'context', 'contract', 'create', 'data', 'define', 'derive',
                'distribute', 'economy', 'environment', 'establish', 'estimate',
                'evidence', 'export', 'factor', 'finance', 'formula', 'function',
                'identify', 'income', 'indicate', 'individual', 'interpret',
                'involve', 'issue', 'labor', 'legal', 'legislate', 'major',
                'method', 'occur', 'percent', 'period', 'policy', 'principle',
                'proceed', 'process', 'require', 'research', 'respond', 'role',
                'section', 'sector', 'significant', 'similar', 'source', 'specific',
                'structure', 'theory', 'vary'
            }
            self._save_word_list(academic_words, academic_path)
    
    def _create_basic_dictionary(self) -> Set[str]:
        """Create a basic English dictionary."""
        # This is a minimal dictionary - in practice, you'd want to use
        # a comprehensive word list like NLTK's words corpus
        basic_words = {
            # Common words
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us'
        }
        return basic_words
    
    def _create_basic_frequencies(self) -> Dict[str, int]:
        """Create basic word frequency dictionary."""
        # Sample frequencies - in practice, use corpus-derived frequencies
        return {
            'the': 1000000, 'be': 800000, 'to': 700000, 'of': 600000,
            'and': 500000, 'a': 450000, 'in': 400000, 'that': 350000,
            'have': 300000, 'i': 280000, 'it': 260000, 'for': 240000,
            'not': 220000, 'on': 200000, 'with': 180000, 'he': 160000,
            'as': 140000, 'you': 130000, 'do': 120000, 'at': 110000
        }