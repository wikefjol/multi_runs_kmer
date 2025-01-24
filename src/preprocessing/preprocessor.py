from typing import Protocol, List, Any
from src.utils.logging_utils import with_logging
from src.utils.vocab import Vocabulary
class Strategy(Protocol):
    '''Augments sequence by imitating sequencing errors'''
    def execute(self, sequence: list[str]) -> list[str]:
        """
        Parameters
        ----------
        sequence : str
            DNA sequence

        Returns
        ----------
        sequence : str
            Augmented DNA sequence
        """

class Preprocessor:
    def __init__(
        self,
        augmentation_strategy: Strategy,
        tokenization_strategy: Strategy,
        padding_strategy: Strategy,
        truncation_strategy: Strategy,
        vocab: Vocabulary = None
    ):
        self.augmentation_strategy = augmentation_strategy
        self.tokenization_strategy = tokenization_strategy
        self.padding_strategy = padding_strategy
        self.truncation_strategy = truncation_strategy
        self.vocab = vocab

    @with_logging(level=10)
    def process(self, sequence: str) -> List[List[str]]:
        sequence = list(sequence)  # Convert string to list of characters
    
        augmented_sequence: List[str] = self.augmentation_strategy.execute(sequence)
        tokenized_sentence: List[List[str]] = self.tokenization_strategy.execute(augmented_sequence)
        padded_sentence: List[List[str]] = self.padding_strategy.execute(tokenized_sentence)
        processed_sentence: List[List[str]] = self.truncation_strategy.execute(padded_sentence)
        mapped_sentence: List[List[int]] = self.vocab.map_sentence(processed_sentence)
        
        return mapped_sentence