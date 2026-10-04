from dataclasses import dataclass


@dataclass
class DefaultStrategy:
    absgrad: bool = False
    verbose: bool = False
