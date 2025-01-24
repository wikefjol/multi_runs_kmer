import importlib

from inspect import signature
from typing import Any, Protocol


from src.preprocessing.augmentation import SequenceModifier
from src.preprocessing.preprocessor import Preprocessor
from src.utils.errors import ConstructionError, StrategyError
from src.utils.logging_utils import with_logging



class Modifier(Protocol):
    """Protocol defining sequence modification methods."""
    alphabet: list[str]

    def _insert(self, seq: list[str], idx: int) -> None: pass
    def _replace(self, seq: list[str], idx: int) -> None: pass
    def _delete(self, seq: list[str], idx: int) -> None: pass
    def _swap(self, seq: list[str], idx: int) -> None: pass


@with_logging(level=8)
def load_strategy_module(strategy_type: str) -> Any:
    """Dynamically load and return the strategy module."""
    full_module_name = f"src.preprocessing.{strategy_type}"
    try:
        return importlib.import_module(full_module_name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{full_module_name}' could not be imported.")


@with_logging(level=8)
def prepare_strategy(module: Any, class_name: str, **kwargs) -> Any:
    """Validate and instantiate the strategy class from the module."""
    try:
        strategy_class = getattr(module, class_name)
    except AttributeError:
        available_classes = [attr for attr in dir(module) if not attr.startswith("_")]
        raise StrategyError(
            f"Class '{class_name}' not found in module '{module.__name__}'. "
            f"Available classes: {available_classes}"
        )

    init_signature = signature(strategy_class)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_signature.parameters}

    missing_args = [
        param.name for param in init_signature.parameters.values()
        if param.default == param.empty and param.name not in filtered_kwargs
    ]
    if missing_args:
        raise ValueError(f"Missing required arguments for {strategy_class.__name__}: {missing_args}")

    return strategy_class(**filtered_kwargs)


@with_logging(level=9)
def get_strategy(strategy_type: str, **kwargs) -> Any:
    """Dynamically load and return an instance of a strategy class."""
    strategy_name = kwargs.pop("strategy", None)
    if not strategy_name:
        raise ValueError(f"Missing 'strategy' in configuration for {strategy_type}.")
    
    class_name = strategy_name.capitalize() + "Strategy"
    module = load_strategy_module(strategy_type)
    return prepare_strategy(module, class_name, **kwargs)

@with_logging(level=20)
def create_preprocessor(config, vocab, training = True) -> Preprocessor:
    """Create and return a Preprocessor instance based on the configuration."""
    alphabet = vocab.get_alphabet()

    try:
        if training:
            augmentation_config = config["augmentation"]["training"]

        else: #if not training it has to be eval
            augmentation_config = config["augmentation"]["evaluation"]

        tokenization_config = config["tokenization"]
        padding_config = config["padding"]
        truncation_config = config["truncation"]
        
        #augmentation_config["alphabet"] = alphabet        
        #tokenization_config["alphabet"] = alphabet        
        
        # print("augmentation config", augmentation_config)
        # print("tokenization config", tokenization_config)
        # print("padding config",padding_config)
        # print("truncation config",truncation_config)
        
        # Create the modifier instance
        modifier = SequenceModifier(alphabet)

        # Get strategies
        augmentation_strategy = get_strategy(strategy_type = "augmentation", modifier = modifier, **augmentation_config)
        tokenization_strategy = get_strategy(strategy_type = "tokenization", **tokenization_config)
        padding_strategy      = get_strategy(strategy_type = "padding", **padding_config)
        truncation_strategy   = get_strategy(strategy_type = "truncation", **truncation_config)

    except KeyError as e:
        raise ConstructionError(f"Strategy configuration error: {e}")
    except StrategyError as e:
        raise ConstructionError(f"Error in strategy setup: {e}")

    return Preprocessor(
        augmentation_strategy = augmentation_strategy,
        tokenization_strategy = tokenization_strategy,
        padding_strategy = padding_strategy,
        truncation_strategy = truncation_strategy,
        vocab = vocab,
    )