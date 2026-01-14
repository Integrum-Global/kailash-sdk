# Phase 1B: Build-Time Validation Specification
**Weeks 5-6, $8K, 2 Developers**

## Executive Summary

Phase 1B introduces build-time validation to catch 80% of DataFlow configuration errors BEFORE runtime execution, eliminating costly debugging cycles.

**Goal**: Shift error discovery from runtime (hours of debugging) to registration time (immediate feedback).

**Components:**
1. Enhanced `@db.model` decorator with validation
2. CLI validator tool (`dataflow validate`)
3. Error-to-solution knowledge base

**Success Criteria:**
- 80% of common errors caught at registration time
- Validation overhead <100ms per model
- Zero false positives in warning mode
- <5% false positives in strict mode

---

## Component 1: Enhanced @db.model Decorator

### File: `src/dataflow/decorators.py`

**Current State** (Line 1-150):
```python
def model(cls):
    """Register model with DataFlow - minimal validation"""
    # Basic registration only
    return cls
```

**Enhanced State** (Add 200 LOC):
```python
from enum import Enum
from typing import Optional, List
import warnings

class ValidationMode(Enum):
    """Validation strictness levels"""
    OFF = "off"          # No validation (backward compat)
    WARN = "warn"        # Validate, log warnings (default)
    STRICT = "strict"    # Validate, raise errors

class ValidationResult:
    """Result of model validation"""
    def __init__(self):
        self.is_valid = True
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationWarning] = []

class ValidationError:
    """Validation error details"""
    def __init__(self, code: str, message: str, field: str = None):
        self.code = code
        self.message = message
        self.field = field

class ValidationWarning:
    """Validation warning details"""
    def __init__(self, code: str, message: str, field: str = None):
        self.code = code
        self.message = message
        self.field = field

def model(
    cls = None,
    *,
    validation: ValidationMode = ValidationMode.WARN,
    strict: bool = None,  # Shorthand for validation=STRICT
    skip_validation: bool = False  # Opt-out for advanced users
):
    """
    Enhanced model decorator with build-time validation.

    Args:
        cls: Model class to decorate
        validation: Validation mode (OFF, WARN, STRICT)
        strict: Shorthand for validation=STRICT
        skip_validation: Skip all validation (for advanced users)

    Usage:
        # Default: Warning mode
        @db.model
        class User(Base):
            ...

        # Strict mode
        @db.model(strict=True)
        class User(Base):
            ...

        # Custom validation mode
        @db.model(validation=ValidationMode.OFF)
        class User(Base):
            ...
    """
    def decorator(cls):
        # Determine validation mode
        if skip_validation:
            mode = ValidationMode.OFF
        elif strict is not None:
            mode = ValidationMode.STRICT if strict else ValidationMode.WARN
        else:
            mode = validation

        # Run validation
        if mode != ValidationMode.OFF:
            result = _validate_model_schema(cls, mode)

            # Handle validation results
            if mode == ValidationMode.STRICT and not result.is_valid:
                raise ModelValidationError(result.errors)
            elif mode == ValidationMode.WARN:
                for warning in result.warnings:
                    warnings.warn(
                        f"[{warning.code}] {warning.message}",
                        DataFlowValidationWarning
                    )
                for error in result.errors:
                    warnings.warn(
                        f"[{error.code}] POTENTIAL ERROR: {error.message}",
                        DataFlowValidationWarning
                    )

        # Proceed with registration
        # ... existing registration logic ...

        return cls

    if cls is None:
        # Called with parameters: @db.model(strict=True)
        return decorator
    else:
        # Called without parameters: @db.model
        return decorator(cls)
```

### Validation Logic: `_validate_model_schema()`

**Add to `src/dataflow/decorators.py`** (150 LOC):

```python
def _validate_model_schema(cls, mode: ValidationMode) -> ValidationResult:
    """
    Validate model schema for common issues.

    Checks:
    1. Primary key existence and naming
    2. Auto-managed field conflicts
    3. SQLAlchemy type validity
    4. Field naming conventions
    5. Relationship configurations
    """
    result = ValidationResult()

    # Check 1: Primary key validation
    result = _validate_primary_key(cls, result)

    # Check 2: Auto-managed fields
    result = _validate_auto_managed_fields(cls, result)

    # Check 3: Field types
    result = _validate_field_types(cls, result)

    # Check 4: Naming conventions
    result = _validate_naming_conventions(cls, result)

    # Check 5: Relationships
    result = _validate_relationships(cls, result)

    return result

def _validate_primary_key(cls, result: ValidationResult) -> ValidationResult:
    """Validate primary key exists and follows convention."""
    if not hasattr(cls, '__table__'):
        result.errors.append(ValidationError(
            code="VAL-001",
            message=f"Model {cls.__name__} has no __table__ attribute"
        ))
        result.is_valid = False
        return result

    table = cls.__table__
    pk_columns = [col for col in table.columns if col.primary_key]

    if not pk_columns:
        result.errors.append(ValidationError(
            code="VAL-002",
            message=f"Model {cls.__name__} has no primary key defined"
        ))
        result.is_valid = False
        return result

    # Check primary key naming convention
    if len(pk_columns) == 1:
        pk_name = pk_columns[0].name
        if pk_name != 'id':
            result.warnings.append(ValidationWarning(
                code="VAL-003",
                message=(
                    f"Primary key '{pk_name}' should be named 'id' for DataFlow compatibility. "
                    f"Current naming may cause node generation issues."
                ),
                field=pk_name
            ))
    else:
        result.warnings.append(ValidationWarning(
            code="VAL-004",
            message=f"Model {cls.__name__} has composite primary key. This may complicate node operations."
        ))

    return result

def _validate_auto_managed_fields(cls, result: ValidationResult) -> ValidationResult:
    """Check for auto-managed field conflicts."""
    table = cls.__table__
    auto_managed_fields = ['created_at', 'updated_at', 'created_by', 'updated_by']

    for field in auto_managed_fields:
        if field in [col.name for col in table.columns]:
            result.warnings.append(ValidationWarning(
                code="VAL-005",
                message=(
                    f"Field '{field}' in {cls.__name__} may conflict with auto-managed fields. "
                    f"If using enable_audit=True, this field will be auto-populated. "
                    f"Consider using a different name or ensure enable_audit=False."
                ),
                field=field
            ))

    return result

def _validate_field_types(cls, result: ValidationResult) -> ValidationResult:
    """Validate SQLAlchemy field types are compatible."""
    table = cls.__table__

    # Check for problematic types
    for col in table.columns:
        col_type = str(col.type)

        # Check for datetime without timezone
        if 'DATETIME' in col_type.upper() and 'timezone' not in col_type.lower():
            result.warnings.append(ValidationWarning(
                code="VAL-006",
                message=(
                    f"Field '{col.name}' uses DateTime without timezone. "
                    f"Consider using DateTime(timezone=True) to avoid timezone issues."
                ),
                field=col.name
            ))

        # Check for Text without length
        if 'TEXT' in col_type.upper() or 'VARCHAR' in col_type.upper():
            if not hasattr(col.type, 'length') or col.type.length is None:
                result.warnings.append(ValidationWarning(
                    code="VAL-007",
                    message=(
                        f"Field '{col.name}' has no length constraint. "
                        f"Consider adding length for validation and database optimization."
                    ),
                    field=col.name
                ))

    return result

def _validate_naming_conventions(cls, result: ValidationResult) -> ValidationResult:
    """Validate field naming follows conventions."""
    table = cls.__table__

    for col in table.columns:
        # Check for camelCase (should be snake_case)
        if any(c.isupper() for c in col.name):
            result.warnings.append(ValidationWarning(
                code="VAL-008",
                message=(
                    f"Field '{col.name}' uses camelCase. "
                    f"Python convention is snake_case for consistency."
                ),
                field=col.name
            ))

        # Check for reserved words
        reserved_words = ['type', 'class', 'def', 'return', 'id', 'metadata']
        if col.name in reserved_words and col.name != 'id':
            result.warnings.append(ValidationWarning(
                code="VAL-009",
                message=(
                    f"Field '{col.name}' is a Python reserved word. "
                    f"Consider renaming to avoid potential conflicts."
                ),
                field=col.name
            ))

    return result

def _validate_relationships(cls, result: ValidationResult) -> ValidationResult:
    """Validate relationship configurations."""
    # Check for relationships
    if hasattr(cls, '__mapper__'):
        mapper = cls.__mapper__
        for rel in mapper.relationships:
            # Check backref naming
            if hasattr(rel, 'backref') and rel.backref:
                # Validate backref doesn't conflict
                pass

            # Check cascade settings
            if 'delete' in str(rel.cascade):
                result.warnings.append(ValidationWarning(
                    code="VAL-010",
                    message=(
                        f"Relationship '{rel.key}' has delete cascade. "
                        f"Ensure this is intentional to avoid data loss."
                    )
                ))

    return result
```

### Custom Exceptions

**Add to `src/dataflow/exceptions.py`**:

```python
class ModelValidationError(Exception):
    """Raised when model validation fails in strict mode."""
    def __init__(self, errors: List[ValidationError]):
        self.errors = errors
        messages = [f"[{e.code}] {e.message}" for e in errors]
        super().__init__("Model validation failed:\n" + "\n".join(messages))

class DataFlowValidationWarning(UserWarning):
    """Warning category for DataFlow validation issues."""
    pass
```

### Testing Specifications

**File**: `tests/unit/test_model_validation.py` (600 LOC)

**Test Cases** (20 total):

```python
import pytest
from dataflow import DataFlow
from dataflow.decorators import ValidationMode, ModelValidationError
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TestPrimaryKeyValidation:
    """Test primary key validation rules."""

    def test_model_without_primary_key_strict_mode(self):
        """Should raise error in strict mode."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.raises(ModelValidationError) as exc_info:
            @db.model(strict=True)
            class User(Base):
                __tablename__ = "users"
                name = Column(String(100))

        assert "VAL-002" in str(exc_info.value)
        assert "no primary key" in str(exc_info.value).lower()

    def test_model_without_primary_key_warn_mode(self):
        """Should warn in default mode."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model  # Default: warn mode
            class User(Base):
                __tablename__ = "users"
                name = Column(String(100))

        assert any("VAL-002" in str(w.message) for w in record)

    def test_primary_key_not_named_id_warning(self):
        """Should warn if primary key isn't 'id'."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class User(Base):
                __tablename__ = "users"
                user_id = Column(Integer, primary_key=True)
                name = Column(String(100))

        warnings = [str(w.message) for w in record]
        assert any("VAL-003" in w and "should be named 'id'" in w for w in warnings)

    def test_composite_primary_key_warning(self):
        """Should warn about composite primary keys."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning):
            @db.model
            class UserRole(Base):
                __tablename__ = "user_roles"
                user_id = Column(Integer, primary_key=True)
                role_id = Column(Integer, primary_key=True)

        # Should complete without error but with warning

class TestAutoManagedFieldValidation:
    """Test auto-managed field conflict detection."""

    def test_created_at_field_warning(self):
        """Should warn about created_at field."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class User(Base):
                __tablename__ = "users"
                id = Column(Integer, primary_key=True)
                created_at = Column(DateTime)  # Conflicts with auto-managed

        warnings = [str(w.message) for w in record]
        assert any("VAL-005" in w and "created_at" in w for w in warnings)

    def test_updated_at_field_warning(self):
        """Should warn about updated_at field."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class User(Base):
                __tablename__ = "users"
                id = Column(Integer, primary_key=True)
                updated_at = Column(DateTime)

        warnings = [str(w.message) for w in record]
        assert any("VAL-005" in w and "updated_at" in w for w in warnings)

class TestFieldTypeValidation:
    """Test field type validation."""

    def test_datetime_without_timezone_warning(self):
        """Should warn about DateTime without timezone."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class Event(Base):
                __tablename__ = "events"
                id = Column(Integer, primary_key=True)
                event_time = Column(DateTime)  # No timezone

        warnings = [str(w.message) for w in record]
        assert any("VAL-006" in w and "timezone" in w for w in warnings)

    def test_text_without_length_warning(self):
        """Should warn about text fields without length."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class Article(Base):
                __tablename__ = "articles"
                id = Column(Integer, primary_key=True)
                content = Column(String)  # No length

        warnings = [str(w.message) for w in record]
        assert any("VAL-007" in w and "length" in w for w in warnings)

class TestNamingConventionValidation:
    """Test naming convention validation."""

    def test_camelcase_field_warning(self):
        """Should warn about camelCase field names."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class User(Base):
                __tablename__ = "users"
                id = Column(Integer, primary_key=True)
                firstName = Column(String(100))  # camelCase

        warnings = [str(w.message) for w in record]
        assert any("VAL-008" in w and "camelCase" in w for w in warnings)

    def test_reserved_word_field_warning(self):
        """Should warn about reserved word field names."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning) as record:
            @db.model
            class Document(Base):
                __tablename__ = "documents"
                id = Column(Integer, primary_key=True)
                type = Column(String(50))  # Reserved word

        warnings = [str(w.message) for w in record]
        assert any("VAL-009" in w and "reserved word" in w for w in warnings)

class TestValidationModes:
    """Test different validation modes."""

    def test_skip_validation_mode(self):
        """Should skip all validation."""
        db = DataFlow("sqlite:///:memory:")

        # Should not raise or warn
        @db.model(skip_validation=True)
        class User(Base):
            __tablename__ = "users"
            # No primary key - should be ignored
            name = Column(String(100))

        # Should complete without warnings

    def test_off_mode_no_validation(self):
        """OFF mode should skip validation."""
        db = DataFlow("sqlite:///:memory:")

        @db.model(validation=ValidationMode.OFF)
        class User(Base):
            __tablename__ = "users"
            name = Column(String(100))  # No PK

        # Should complete without warnings

    def test_warn_mode_logs_warnings(self):
        """WARN mode should log warnings."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.warns(DataFlowValidationWarning):
            @db.model(validation=ValidationMode.WARN)
            class User(Base):
                __tablename__ = "users"
                name = Column(String(100))  # No PK

    def test_strict_mode_raises_errors(self):
        """STRICT mode should raise errors."""
        db = DataFlow("sqlite:///:memory:")

        with pytest.raises(ModelValidationError):
            @db.model(validation=ValidationMode.STRICT)
            class User(Base):
                __tablename__ = "users"
                name = Column(String(100))  # No PK

class TestPerformance:
    """Test validation performance."""

    def test_validation_overhead_acceptable(self):
        """Validation should add <100ms per model."""
        import time
        db = DataFlow("sqlite:///:memory:")

        start = time.time()

        @db.model
        class User(Base):
            __tablename__ = "users"
            id = Column(Integer, primary_key=True)
            name = Column(String(100))
            email = Column(String(100))
            created_at = Column(DateTime)

        elapsed = time.time() - start
        assert elapsed < 0.1, f"Validation took {elapsed*1000:.2f}ms (should be <100ms)"
```

### Integration with Existing Code

**Modify `src/dataflow/engine.py`** line 89:

```python
# Before
def register_model(self, model_class):
    """Register model with DataFlow."""
    # ... existing registration ...

# After
def register_model(self, model_class, validation_mode=ValidationMode.WARN):
    """Register model with DataFlow."""
    # Run validation if not already done by decorator
    if not hasattr(model_class, '_dataflow_validated'):
        result = _validate_model_schema(model_class, validation_mode)
        if validation_mode == ValidationMode.STRICT and not result.is_valid:
            raise ModelValidationError(result.errors)
        model_class._dataflow_validated = True

    # ... existing registration ...
```

---

## Component 2: CLI Validator Tool

### File: `src/dataflow/cli/validate.py` (300 LOC)

**Purpose**: Standalone validation tool for pre-deployment checks.

**Implementation**:

```python
"""
DataFlow CLI Validator

Usage:
    dataflow validate models.py
    dataflow validate --strict models.py
    dataflow validate --config validation_config.yaml models.py
"""

import argparse
import sys
import importlib.util
from pathlib import Path
from typing import List, Dict
import yaml

from dataflow import DataFlow
from dataflow.decorators import ValidationMode, ValidationResult
from dataflow.core.error_enhancer import ErrorEnhancer

class ValidationConfig:
    """Validation configuration."""
    def __init__(self, config_file: Path = None):
        self.mode = ValidationMode.WARN
        self.ignore_codes: List[str] = []
        self.fail_on_warnings = False

        if config_file and config_file.exists():
            self._load_config(config_file)

    def _load_config(self, config_file: Path):
        """Load configuration from YAML file."""
        with open(config_file) as f:
            config = yaml.safe_load(f)

        self.mode = ValidationMode(config.get('mode', 'warn'))
        self.ignore_codes = config.get('ignore_codes', [])
        self.fail_on_warnings = config.get('fail_on_warnings', False)

class ModelValidator:
    """Validate Python files containing DataFlow models."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results: Dict[str, ValidationResult] = {}

    def validate_file(self, file_path: Path) -> ValidationResult:
        """Validate all models in a Python file."""
        # Import the module
        spec = importlib.util.spec_from_file_location("models", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find all DataFlow model classes
        models = self._find_models(module)

        # Validate each model
        combined_result = ValidationResult()
        for model in models:
            result = self._validate_model(model)
            combined_result.errors.extend(result.errors)
            combined_result.warnings.extend(result.warnings)
            if not result.is_valid:
                combined_result.is_valid = False

        self.results[str(file_path)] = combined_result
        return combined_result

    def _find_models(self, module):
        """Find all model classes in module."""
        models = []
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and
                hasattr(obj, '__table__') and
                hasattr(obj, '_dataflow_model')):
                models.append(obj)
        return models

    def _validate_model(self, model_class):
        """Validate a single model."""
        from dataflow.decorators import _validate_model_schema
        return _validate_model_schema(model_class, self.config.mode)

    def print_report(self):
        """Print validation report."""
        total_errors = sum(len(r.errors) for r in self.results.values())
        total_warnings = sum(len(r.warnings) for r in self.results.values())

        print("\n" + "="*60)
        print("DataFlow Validation Report")
        print("="*60)

        for file_path, result in self.results.items():
            print(f"\nFile: {file_path}")
            print("-" * 60)

            if result.errors:
                print(f"\n❌ Errors ({len(result.errors)}):")
                for error in result.errors:
                    field_info = f" (field: {error.field})" if error.field else ""
                    print(f"  [{error.code}] {error.message}{field_info}")

            if result.warnings:
                print(f"\n⚠️  Warnings ({len(result.warnings)}):")
                for warning in result.warnings:
                    if warning.code not in self.config.ignore_codes:
                        field_info = f" (field: {warning.field})" if warning.field else ""
                        print(f"  [{warning.code}] {warning.message}{field_info}")

            if not result.errors and not result.warnings:
                print("\n✅ No issues found")

        print("\n" + "="*60)
        print(f"Summary: {total_errors} errors, {total_warnings} warnings")
        print("="*60 + "\n")

        # Exit code based on results
        if total_errors > 0:
            return 1
        if self.config.fail_on_warnings and total_warnings > 0:
            return 1
        return 0

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate DataFlow model definitions"
    )
    parser.add_argument(
        'files',
        nargs='+',
        type=Path,
        help="Python files containing models"
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help="Use strict validation mode"
    )
    parser.add_argument(
        '--config',
        type=Path,
        help="Path to validation config file"
    )
    parser.add_argument(
        '--fail-on-warnings',
        action='store_true',
        help="Fail if any warnings are found"
    )

    args = parser.parse_args()

    # Load configuration
    config = ValidationConfig(args.config)
    if args.strict:
        config.mode = ValidationMode.STRICT
    if args.fail_on_warnings:
        config.fail_on_warnings = True

    # Validate files
    validator = ModelValidator(config)
    for file_path in args.files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        validator.validate_file(file_path)

    # Print report and exit with appropriate code
    exit_code = validator.print_report()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
```

### Configuration File Example

**File**: `validation_config.yaml`

```yaml
# DataFlow Validation Configuration

# Validation mode: off, warn, strict
mode: warn

# Codes to ignore (won't be reported)
ignore_codes:
  - VAL-008  # Ignore camelCase warnings
  - VAL-009  # Ignore reserved word warnings

# Fail build if warnings are found
fail_on_warnings: false

# Custom rules (future expansion)
custom_rules:
  - name: require_string_lengths
    description: "All String fields must have explicit length"
    pattern: "String()"
    severity: error
```

### Integration with CI/CD

**File**: `.github/workflows/validate-dataflow.yml`

```yaml
name: DataFlow Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install kailash-dataflow

      - name: Validate models
        run: |
          dataflow validate --strict src/models.py
```

### Testing CLI Tool

**File**: `tests/cli/test_validate_cli.py` (400 LOC)

```python
import pytest
from pathlib import Path
import tempfile
import subprocess

class TestValidateCLI:
    """Test CLI validator tool."""

    def test_validate_valid_model(self, tmp_path):
        """Should pass validation for valid model."""
        model_file = tmp_path / "models.py"
        model_file.write_text("""
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
""")

        result = subprocess.run(
            ["dataflow", "validate", str(model_file)],
            capture_output=True
        )

        assert result.returncode == 0
        assert "No issues found" in result.stdout.decode()

    def test_validate_invalid_model_strict(self, tmp_path):
        """Should fail validation for invalid model in strict mode."""
        model_file = tmp_path / "models.py"
        model_file.write_text("""
from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    name = Column(String(100))  # No primary key
""")

        result = subprocess.run(
            ["dataflow", "validate", "--strict", str(model_file)],
            capture_output=True
        )

        assert result.returncode == 1
        assert "VAL-002" in result.stdout.decode()
        assert "no primary key" in result.stdout.decode().lower()

    def test_validate_with_config(self, tmp_path):
        """Should use configuration file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
mode: warn
ignore_codes:
  - VAL-003
""")

        model_file = tmp_path / "models.py"
        model_file.write_text("""
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)  # Should warn VAL-003
    name = Column(String(100))
""")

        result = subprocess.run(
            ["dataflow", "validate", "--config", str(config_file), str(model_file)],
            capture_output=True
        )

        # Should pass (warning ignored)
        assert result.returncode == 0
```

---

## Component 3: Error-to-Solution Knowledge Base

### File: `src/dataflow/errors/knowledge_base.yaml` (1,000 lines)

**Structure**:

```yaml
# DataFlow Error-to-Solution Knowledge Base
# Maps error patterns to actionable solutions

errors:
  - code: DF-101
    pattern: "Parameter.*data.*missing"
    category: parameter
    severity: high
    contexts:
      - node_type: CreateNode
        symptoms:
          - "KeyError: 'data'"
          - "Parameter 'data' missing"
        root_causes:
          - cause: "Connection not established"
            probability: 60%
            diagnostic: "Check workflow.connections for missing connection to this node"
          - cause: "Parameter name mismatch"
            probability: 25%
            diagnostic: "Source node output parameter name doesn't match 'data'"
          - cause: "Empty input to workflow"
            probability: 15%
            diagnostic: "Workflow inputs dictionary is empty or missing keys"
        solutions:
          - priority: 1
            description: "Add connection to provide 'data' parameter"
            code_template: |
              workflow.add_connection(
                  source_node_id="{{source}}",
                  source_param="{{source_param}}",
                  target_node_id="{{node_id}}",
                  target_param="data"
              )
            variables:
              source: "Name of previous node that has the data"
              source_param: "Output parameter name from source node"
              node_id: "Current node ID ({{node_id}})"
          - priority: 2
            description: "Check source node output structure"
            code_template: |
              # Inspect source node outputs
              from dataflow.platform import Inspector
              inspector = Inspector(db)
              node_info = inspector.node("{{source}}")
              print(node_info.output_params)
          - priority: 3
            description: "Verify workflow inputs"
            code_template: |
              # When executing workflow, ensure inputs contain data
              results, run_id = runtime.execute(
                  workflow.build(),
                  inputs={"user_data": {"name": "John", "email": "john@example.com"}}
              )
        examples:
          - scenario: "Basic CRUD workflow"
            before: |
              workflow = WorkflowBuilder()
              workflow.add_node(UserCreateNode, "create_user", {})
              # Missing: connection to provide data
            after: |
              workflow = WorkflowBuilder()
              workflow.add_node(UserCreateNode, "create_user", {})
              workflow.add_connection("input", "user_data", "create_user", "data")
        time_saved: "2-4 hours"

  - code: DF-102
    pattern: "Parameter.*type.*mismatch"
    category: parameter
    severity: high
    contexts:
      - node_type: CreateNode
        symptoms:
          - "TypeError: expected dict, got str"
          - "Invalid parameter type"
        root_causes:
          - cause: "Passing individual fields instead of dict"
            probability: 70%
            diagnostic: "Connections pass field values directly instead of wrapping in dict"
          - cause: "Wrong parameter passed from previous node"
            probability: 20%
            diagnostic: "Source node output type doesn't match expected input type"
          - cause: "JSON serialization issue"
            probability: 10%
            diagnostic: "Data was serialized/deserialized incorrectly"
        solutions:
          - priority: 1
            description: "Wrap fields in dictionary for CreateNode"
            code_template: |
              # CreateNode expects: {'data': {'field1': value1, 'field2': value2}}
              # NOT: {'data': value} or {'field1': value1, 'field2': value2}

              # Correct pattern:
              workflow.add_connection("source", "output_dict", "create_user", "data")

              # If source provides flat fields, use PythonCode to wrap:
              workflow.add_node(PythonCodeNode, "wrap_data", {
                  "code": '''
def execute(name, email):
    return {"data": {"name": name, "email": email}}
'''
              })
              workflow.add_connection("input", "name", "wrap_data", "name")
              workflow.add_connection("input", "email", "wrap_data", "email")
              workflow.add_connection("wrap_data", "data", "create_user", "data")
        examples:
          - scenario: "CreateNode with individual field connections"
            before: |
              # WRONG: Passing fields individually
              workflow.add_connection("input", "name", "create_user", "data.name")
              workflow.add_connection("input", "email", "create_user", "data.email")
            after: |
              # CORRECT: Wrap in dict first
              workflow.add_node(PythonCodeNode, "prepare_data", {
                  "code": "def execute(name, email): return {'name': name, 'email': email}"
              })
              workflow.add_connection("input", "name", "prepare_data", "name")
              workflow.add_connection("input", "email", "prepare_data", "email")
              workflow.add_connection("prepare_data", "return", "create_user", "data")
        time_saved: "1.5-3 hours"

  - code: DF-103
    pattern: "Field.*invalid.*value"
    category: parameter
    severity: medium
    contexts:
      - field_type: "DateTime"
        symptoms:
          - "AttributeError: 'str' object has no attribute 'isoformat'"
          - "datetime serialization error"
        root_causes:
          - cause: "Passing string instead of datetime object"
            probability: 80%
            diagnostic: "Check if datetime fields are datetime objects, not strings"
          - cause: "Timezone-aware vs timezone-naive mismatch"
            probability: 15%
            diagnostic: "Mixing timezone-aware and naive datetimes"
          - cause: "Invalid datetime format"
            probability: 5%
            diagnostic: "String format doesn't match expected datetime format"
        solutions:
          - priority: 1
            description: "Convert strings to datetime objects"
            code_template: |
              from datetime import datetime

              # In your data preparation:
              data = {
                  "name": "John",
                  "created_at": datetime.now()  # datetime object, not string
              }

              # Or parse from string:
              from dateutil import parser
              data = {
                  "name": "John",
                  "created_at": parser.parse("2024-01-01T12:00:00")
              }
          - priority: 2
            description: "Use DataFlow auto-managed fields instead"
            code_template: |
              # Remove created_at/updated_at from model fields
              # Enable auto-management:
              db = DataFlow(
                  database_url="...",
                  enable_audit=True  # Automatically manages created_at/updated_at
              )

              # Don't pass these fields in data:
              data = {"name": "John"}  # created_at added automatically
        time_saved: "0.5-1.5 hours"

# Continue for all 50+ common errors...
```

### Loading Knowledge Base

**Add to `src/dataflow/core/error_enhancer.py`**:

```python
import yaml
from pathlib import Path

class KnowledgeBase:
    """Error-to-solution knowledge base."""

    def __init__(self):
        self.errors = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        """Load error patterns and solutions from YAML."""
        kb_path = Path(__file__).parent.parent / "errors" / "knowledge_base.yaml"
        with open(kb_path) as f:
            data = yaml.safe_load(f)

        for error in data['errors']:
            self.errors[error['code']] = error

    def find_solution(self, error_code: str, context: dict = None):
        """Find solutions for a specific error code."""
        if error_code not in self.errors:
            return None

        error_entry = self.errors[error_code]

        # Find matching context
        if context and 'contexts' in error_entry:
            for ctx in error_entry['contexts']:
                if self._context_matches(ctx, context):
                    return ctx

        # Return first context if no match
        if 'contexts' in error_entry:
            return error_entry['contexts'][0]

        return None

    def _context_matches(self, ctx_definition, actual_context):
        """Check if context matches definition."""
        # Simple matching for now, can be enhanced
        for key, value in ctx_definition.items():
            if key in actual_context and actual_context[key] == value:
                return True
        return False
```

---

## Integration Testing

### File: `tests/integration/test_phase_1b_integration.py` (800 LOC)

**Test comprehensive integration of all Phase 1B components:**

```python
import pytest
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from dataflow import DataFlow
from dataflow.decorators import ValidationMode
from dataflow.cli.validate import ModelValidator, ValidationConfig

Base = declarative_base()

class TestPhase1BIntegration:
    """Integration tests for Phase 1B validation system."""

    def test_end_to_end_strict_validation(self):
        """Test complete flow in strict mode."""
        db = DataFlow("sqlite:///:memory:")

        # Should raise on invalid model
        with pytest.raises(ModelValidationError):
            @db.model(strict=True)
            class User(Base):
                __tablename__ = "users"
                name = Column(String(100))  # No PK

        # Should succeed on valid model
        @db.model(strict=True)
        class ValidUser(Base):
            __tablename__ = "users"
            id = Column(Integer, primary_key=True)
            name = Column(String(100))

        # Verify node generation works
        assert hasattr(db, 'UserCreateNode')
        assert hasattr(db, 'UserReadNode')

    def test_cli_validation_integration(self, tmp_path):
        """Test CLI validator on actual model files."""
        # Create test model file
        model_file = tmp_path / "models.py"
        model_file.write_text("""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)  # Should warn
    name = Column(String(100))
    created_at = Column(DateTime)  # Should warn about auto-managed
""")

        # Run validator
        config = ValidationConfig()
        validator = ModelValidator(config)
        result = validator.validate_file(model_file)

        # Check warnings were generated
        assert len(result.warnings) >= 2
        warning_codes = [w.code for w in result.warnings]
        assert "VAL-003" in warning_codes  # PK naming
        assert "VAL-005" in warning_codes  # Auto-managed field

    def test_knowledge_base_integration(self):
        """Test knowledge base provides solutions."""
        from dataflow.core.error_enhancer import KnowledgeBase

        kb = KnowledgeBase()

        # Find solution for common error
        solution = kb.find_solution("DF-101", {"node_type": "CreateNode"})

        assert solution is not None
        assert 'solutions' in solution
        assert len(solution['solutions']) >= 1
        assert 'code_template' in solution['solutions'][0]
```

---

## Success Criteria & Validation

### Week 5 (Midpoint Check):

**Metrics:**
- [ ] Decorator validation logic complete (200 LOC)
- [ ] 10 validation rules implemented
- [ ] <100ms validation overhead per model
- [ ] Unit tests passing (20 tests)

**Go/No-Go:**
- GO if: Validation catches ≥5 error types, performance acceptable
- NO-GO if: Validation overhead >200ms, <50% tests passing

### Week 6 (Phase 1B Complete):

**Metrics:**
- [ ] All components implemented (500 total LOC)
- [ ] CLI tool functional
- [ ] Knowledge base covers 50+ errors
- [ ] Integration tests passing (15 tests)
- [ ] 80% of common errors caught at registration

**Validation Test:**
1. Create 10 models with known issues
2. Run validation in strict mode
3. Verify 8/10 errors caught before runtime

**Go/No-Go Decision:**
- GO to Phase 1C if: ≥80% error detection, CLI works, performance acceptable
- STOP if: <60% error detection, major performance issues

---

## Risk Mitigation

### Risk 1: Performance Overhead
**Probability**: MEDIUM (30%)
**Mitigation**:
- Lazy validation (only on decorator)
- Cache validation results
- Skip validation for known-good models

### Risk 2: False Positives
**Probability**: HIGH (50%)
**Mitigation**:
- Warning mode by default (not strict)
- Allow ignore codes in configuration
- Community feedback on validation rules

### Risk 3: Integration Breaks
**Probability**: LOW (15%)
**Mitigation**:
- Extensive backward compatibility tests
- Feature flag for validation
- Gradual rollout

---

## Next Steps After Phase 1B

**Proceed to Phase 1C** (Weeks 7-10):
- Enhanced error messages in core
- Strict mode refinement
- AI debugging agent integration

**Success ensures**:
- 80% of errors caught early
- Faster IT team onboarding
- Reduced debugging tokens

---

**Document Status**: IMPLEMENTATION-READY
**Last Updated**: 2025-10-29
**Next Review**: Week 5 (midpoint validation)
