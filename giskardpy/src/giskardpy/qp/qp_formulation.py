from dataclasses import dataclass, field


@dataclass
class QPFormulation:
    has_explicit_acc_variables: bool = field(default=False)
    has_explicit_jerk_variables: bool = field(default=True)
