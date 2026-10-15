from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


SCHEMAS_DIR = Path(__file__).resolve().parent / "schemas"


@dataclass(frozen=True)
class InstanceValidationError:
    path: str
    message: str


def schema_files() -> list[Path]:
    return sorted(SCHEMAS_DIR.glob("*.schema.json"))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_schema(path_or_name: str | Path) -> dict[str, Any]:
    path = Path(path_or_name)
    if not path.is_absolute():
        path = SCHEMAS_DIR / path
    return load_json(path)


def check_schema_file(path: Path) -> None:
    Draft202012Validator.check_schema(load_json(path))


def check_all_schemas(paths: Iterable[Path] | None = None) -> list[Path]:
    checked: list[Path] = []
    for path in paths or schema_files():
        check_schema_file(path)
        checked.append(path)
    return checked


@lru_cache(maxsize=1)
def schema_registry() -> Registry:
    registry = Registry()
    resources = []
    for path in schema_files():
        schema = load_json(path)
        schema_id = schema.get("$id")
        if schema_id:
            resources.append((schema_id, Resource.from_contents(schema, DRAFT202012)))
    return registry.with_resources(resources)


@lru_cache(maxsize=None)
def validator_for(schema_name: str) -> Draft202012Validator:
    """Build each named validator once per process."""
    return Draft202012Validator(load_schema(schema_name), registry=schema_registry())


def validation_errors(instance: Any, schema_name: str) -> list[InstanceValidationError]:
    validator = validator_for(schema_name)
    errors: list[InstanceValidationError] = []
    for error in sorted(validator.iter_errors(instance), key=lambda err: list(err.path)):
        path = "$"
        if error.path:
            path += "".join(f"[{part!r}]" for part in error.path)
        errors.append(InstanceValidationError(path=path, message=error.message))
    return errors


def validate_instance(instance: Any, schema_name: str) -> None:
    errors = validation_errors(instance, schema_name)
    if errors:
        detail = "; ".join(f"{err.path}: {err.message}" for err in errors)
        raise ValidationError(detail)
