from enum import Enum
from typing import Optional, Union, List
from typing_extensions import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# -----------------------------
# Constants
# -----------------------------

MIN_DISTANCE_M = 0.2
MAX_DISTANCE_M = 5.0

MIN_ANGLE_DEG = 30.0
MAX_ANGLE_DEG = 180.0

MIN_SPEED_MPS = 0.05
MAX_SPEED_MPS = 1.0


# -----------------------------
# Enums
# -----------------------------

class OutputStatus(str, Enum):
    OK = "ok"
    CLARIFICATION_REQUIRED = "clarification_required"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"


class CommandAction(str, Enum):
    ADD = "add"
    # UPDATE = "update"
    # REMOVE = "remove"


class ConstraintType(str, Enum):
    SPATIAL = "spatial"
    BEHAVIOR = "behavior"


class SpatialMode(str, Enum):
    AVOID = "avoid"
    REMAIN = "remain"


class BehaviorMode(str, Enum):
    ACTIVATE = "activate"


class RegionConstructor(str, Enum):
    BUFFER = "buffer"
    DIRECTIONAL = "directional"
    BETWEEN = "between"
    WORKSPACE = "workspace"
    DENSITY = "density"
    PREDICTED_OCCUPANCY = "predicted_occupancy"


class DirectionalRelation(str, Enum):
    FRONT = "front"
    BEHIND = "behind"
    LEFT = "left"
    RIGHT = "right"


class TargetKind(str, Enum):
    CLASS = "class"
    INSTANCE = "instance"
    LABEL = "label"


class BehaviorKind(str, Enum):
    VELOCITY_LIMIT = "velocity_limit"
    HEADING_ALIGN = "heading_align"


class LifetimeType(str, Enum):
    PERSISTENT = "persistent"
    DURATION = "duration"


# -----------------------------
# Target models
# -----------------------------

class ClassTarget(BaseModel):
    kind: Literal[TargetKind.CLASS] = TargetKind.CLASS
    class_name: str = Field(alias="class")


class InstanceTarget(BaseModel):
    kind: Literal[TargetKind.INSTANCE] = TargetKind.INSTANCE
    id: str


class LabelTarget(BaseModel):
    kind: Literal[TargetKind.LABEL] = TargetKind.LABEL
    label: str


Target = Annotated[
    Union[ClassTarget, InstanceTarget, LabelTarget],
    Field(discriminator="kind"),
]


# -----------------------------
# Lifetime
# -----------------------------

class PersistentLifetime(BaseModel):
    type: Literal[LifetimeType.PERSISTENT] = LifetimeType.PERSISTENT


class DurationLifetime(BaseModel):
    type: Literal[LifetimeType.DURATION] = LifetimeType.DURATION
    seconds: float = Field(gt=0.0, le=3600.0)


Lifetime = Annotated[
    Union[PersistentLifetime, DurationLifetime],
    Field(discriminator="type"),
]


# -----------------------------
# Region constructors
# -----------------------------

class BufferRegion(BaseModel):
    constructor: Literal[RegionConstructor.BUFFER] = RegionConstructor.BUFFER
    target: Target
    distance: float = Field(ge=MIN_DISTANCE_M, le=MAX_DISTANCE_M)


class DirectionalRegion(BaseModel):
    constructor: Literal[RegionConstructor.DIRECTIONAL] = RegionConstructor.DIRECTIONAL
    target: Target
    relation: DirectionalRelation
    distance: float = Field(ge=MIN_DISTANCE_M, le=MAX_DISTANCE_M)
    angle: float = Field(default=120.0, ge=MIN_ANGLE_DEG, le=MAX_ANGLE_DEG)


class BetweenRegion(BaseModel):
    constructor: Literal[RegionConstructor.BETWEEN] = RegionConstructor.BETWEEN
    targets: List[Target] = Field(min_length=2, max_length=2)
    width: float = Field(default=1.5, ge=MIN_DISTANCE_M, le=MAX_DISTANCE_M)


class WorkspaceRegion(BaseModel):
    constructor: Literal[RegionConstructor.WORKSPACE] = RegionConstructor.WORKSPACE
    label: str


class DensityRegion(BaseModel):
    constructor: Literal[RegionConstructor.DENSITY] = RegionConstructor.DENSITY
    target: Target
    radius: float = Field(default=2.0, ge=MIN_DISTANCE_M, le=MAX_DISTANCE_M)
    threshold: int = Field(default=3, ge=2, le=20)


class PredictedOccupancyRegion(BaseModel):
    constructor: Literal[RegionConstructor.PREDICTED_OCCUPANCY] = RegionConstructor.PREDICTED_OCCUPANCY
    target: Target
    horizon: float = Field(default=2.0, gt=0.0, le=10.0)
    probability_threshold: float = Field(default=0.4, gt=0.0, le=1.0)


Region = Annotated[
    Union[
        BufferRegion,
        DirectionalRegion,
        BetweenRegion,
        WorkspaceRegion,
        DensityRegion,
        PredictedOccupancyRegion,
    ],
    Field(discriminator="constructor"),
]


# -----------------------------
# Behavior modifiers
# -----------------------------

class VelocityLimitBehavior(BaseModel):
    kind: Literal[BehaviorKind.VELOCITY_LIMIT] = BehaviorKind.VELOCITY_LIMIT
    max_speed: float = Field(ge=MIN_SPEED_MPS, le=MAX_SPEED_MPS)


class HeadingAlignBehavior(BaseModel):
    kind: Literal[BehaviorKind.HEADING_ALIGN] = BehaviorKind.HEADING_ALIGN
    target: Target
    tolerance: float = Field(default=0.5, gt=0.0, le=3.14)


Behavior = Annotated[
    Union[VelocityLimitBehavior, HeadingAlignBehavior],
    Field(discriminator="kind"),
]


# -----------------------------
# Constraint models
# -----------------------------

class SpatialConstraint(BaseModel):
    id: str
    type: Literal[ConstraintType.SPATIAL] = ConstraintType.SPATIAL
    mode: SpatialMode
    region: Region
    behavior: None = None
    priority: int = Field(default=1, ge=1, le=3)
    enforce: bool = True
    lifetime: Lifetime = Field(default_factory=PersistentLifetime)


class BehaviorConstraint(BaseModel):
    id: str
    type: Literal[ConstraintType.BEHAVIOR] = ConstraintType.BEHAVIOR
    mode: Literal[BehaviorMode.ACTIVATE] = BehaviorMode.ACTIVATE
    activation_region: Region
    behavior: Behavior
    priority: int = Field(default=1, ge=1, le=3)
    enforce: bool = True
    lifetime: Lifetime = Field(default_factory=PersistentLifetime)


Constraint = Annotated[
    Union[SpatialConstraint, BehaviorConstraint],
    Field(discriminator="type"),
]


# -----------------------------
# Command models
# -----------------------------

class AddCommand(BaseModel):
    action: Literal[CommandAction.ADD] = CommandAction.ADD
    constraint: Constraint


# class UpdateCommand(BaseModel):
#     action: Literal[CommandAction.UPDATE] = CommandAction.UPDATE
#     target_constraint_id: str
#     updates: dict


# class RemoveCommand(BaseModel):
#     action: Literal[CommandAction.REMOVE] = CommandAction.REMOVE
#     target_constraint_id: str


# Command = Annotated[
#     Union[AddCommand, UpdateCommand, RemoveCommand],
#     Field(discriminator="action"),
# ]

Command = AddCommand


# -----------------------------
# Top-level output
# -----------------------------

class LLMCompilerOutput(BaseModel):
    status: OutputStatus
    commands: List[Command] = Field(default_factory=list)
    reason: Optional[str] = None
    question: Optional[str] = None

    @model_validator(mode="after")
    def validate_status_consistency(self):
        if self.status == OutputStatus.OK:
            if not self.commands:
                raise ValueError("status='ok' requires at least one command")
            if self.reason is not None:
                raise ValueError("status='ok' should not include a reason")

        if self.status in {
            OutputStatus.CLARIFICATION_REQUIRED,
            OutputStatus.REJECTED,
            OutputStatus.UNSUPPORTED,
        }:
            if self.commands:
                raise ValueError(f"status='{self.status}' must not include commands")
            if not self.reason:
                raise ValueError(f"status='{self.status}' requires a reason")

        if self.status == OutputStatus.CLARIFICATION_REQUIRED:
            if not self.question:
                raise ValueError("clarification_required requires a question")

        return self
