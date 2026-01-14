# Tests Directory - Claude Code Instructions

## 🚨 CRITICAL: Always Use Test Environment Before Running Tests

**NEVER run tests directly without the test environment!** The test environment ensures:
- Correct database setup with proper ports (PostgreSQL on 5434, not 5432)
- Ollama models are available
- Redis is running
- All test data is properly seeded

## 💨 Fast Test Execution

**Run unit tests directly for 11x faster performance!**
- Tests complete in ~10 seconds instead of ~117 seconds
- Proper test isolation through improved fixtures
- 99.96% pass rate with optimized state management
- **Recommended**: `pytest tests/unit/`

### Test Isolation Without Forking

The Kailash SDK now uses smart fixture-based isolation:
1. `isolate_global_state` fixture manages NodeRegistry state
2. Only test-added nodes are cleaned up, SDK nodes preserved
3. Class identities maintained for proper isinstance() checks
4. Module cache preserved to avoid reimport issues

### Tests Requiring Process Isolation

A small number of tests (< 1%) require process isolation due to mock state pollution. These are marked with `@pytest.mark.requires_isolation`:

```python
@pytest.mark.requires_isolation
def test_with_complex_mocking():
    # This test uses mock.patch in ways that pollute global state
    ...
```

**Running isolation tests:**
```bash
# Run only isolation tests (with forking)
pytest -m requires_isolation

# Run all tests except isolation tests (fast mode)
pytest --no-isolation

# Default behavior: runs isolation tests with forking automatically
pytest tests/unit/
```

## 🚀 Test Environment Setup (REQUIRED)

```bash
# From project root:
./test-env setup   # One-time setup (downloads models, initializes databases)
./test-env up      # Start all test services (PostgreSQL, Redis, Ollama)
./test-env test tier1  # Run unit tests (fast mode)
./test-env test tier2  # Run integration tests
./test-env test tier3  # Run E2E tests

# Manual test execution:
pytest tests/unit/ --tb=short           # Fast unit tests (11x faster)
pytest tests/integration/ --tb=short    # Integration tests
pytest tests/e2e/ --tb=short           # E2E tests

# Check if services are running:
./test-env status
```

## 📋 Test Execution Checklist

Before running ANY tests:
1. ✅ Run `./test-env up` to start Docker services
2. ✅ Verify services with `./test-env status`
3. ✅ Use `./test-env test tier<N>` for consistent execution

## 🗄️ Database Configuration

The test environment uses:
- **Host**: localhost
- **Port**: 5434 (NOT 5432!)
- **Database**: kailash_test
- **User**: test_user
- **Password**: test_password

## 🧪 Test Structure

```
tests/
├── unit/          # Tier 1: Fast, no external dependencies
├── integration/   # Tier 2: Component interactions with Docker
├── e2e/          # Tier 3: Full end-to-end scenarios
└── utils/        # Test utilities and Docker configuration
```

## 🎯 TODO-111 Testing Patterns

**Core SDK Architecture Tests**: 67 comprehensive tests for critical infrastructure components:

### Unit Tests (47 tests):
- **CyclicWorkflowExecutor**: 14 tests for `_execute_dag_portion`, `_execute_cycle_groups`, `_propagate_parameters`
- **WorkflowVisualizer**: 14 tests for optional workflow parameter and enhanced methods
- **ConnectionManager**: 19 tests for `filter_events()` and `process_event()` functionality

### Integration Tests (15 tests):
- **Real Docker Infrastructure**: All tests use actual Docker services
- **Component Interactions**: Workflow visualization with real workflows
- **No Mocking Policy**: Integration tests use real SDK components

### E2E Tests (5 tests):
- **Real File I/O**: Tests use actual CSV files and file operations
- **Production Scenarios**: API simulations with realistic timing
- **Complex Workflows**: Multi-cycle workflows with visualization

## ⚠️ Common Issues and Solutions

### "Connection refused" or "Database not found"
- **Solution**: Run `./test-env up` - you forgot to start the test environment

### "Model not found" (Ollama)
- **Solution**: Run `./test-env setup` - models haven't been downloaded

### "Foreign key constraint violation"
- **Solution**: Tests are not using proper test data fixtures from base classes

### "WorkflowConnectionPool is not JSON serializable"
- **Solution**: Use direct asyncpg connections in AsyncPythonCodeNode, not pool objects

## 🔧 E2E Test Infrastructure

For E2E tests, always:
1. Extend `DurableGatewayTestBase` for proper setup/teardown
2. Use `E2ETestConfig.get_async_db_code()` for database operations
3. Use test data helpers (`get_test_customer()`, `create_test_order()`)
4. Never create random IDs - use consistent test data

## 🚨 Node Execution Policy

### Synchronous Nodes
**ALWAYS use `.execute()` to run nodes:**
```python
# ✅ CORRECT
result = node.execute(params)

# ❌ WRONG
result = node.run(params)
result = node.process(params)
result = node.call(params)
```

### AsyncNodes (CRITICAL)
**In async tests, use `.execute_async()`:**
```python
# ✅ CORRECT - Async context
@pytest.mark.asyncio
async def test_async_node():
    node = MyAsyncNode()
    result = await node.execute_async(params)

# ❌ WRONG - Don't use execute() in async context
@pytest.mark.asyncio
async def test_async_node():
    node = MyAsyncNode()
    result = await node.execute(params)  # execute() is not awaitable!

# ❌ WRONG - Don't call run() directly
@pytest.mark.asyncio
async def test_async_node():
    node = MyAsyncNode()
    result = await node.run(params)  # run() doesn't exist on AsyncNode!
```

**NodeParameter Type Requirements:**
```python
# ✅ CORRECT - Always include type
"param": NodeParameter(
    name="param",
    type=str,  # REQUIRED!
    required=True,
    description="A parameter"
)

# ❌ WRONG - Missing type
"param": NodeParameter(
    name="param",
    required=True,
    description="A parameter"
)  # Will fail validation!
```

See [AsyncNode Implementation Guide](../sdk-users/3-development/guides/async-node-guide.md) for full details.

## 📝 Quick Reference

```bash
# Start test environment (ALWAYS DO THIS FIRST!)
./test-env up

# Run specific test tiers
./test-env test tier1    # Unit tests only (fast mode)
./test-env test tier2    # Integration tests
./test-env test tier3    # E2E tests

# Run specific test file
pytest tests/unit/test_specific.py -v    # Fast execution
pytest tests/integration/test_specific.py -v
pytest tests/e2e/test_specific.py -v

# Run with coverage
./test-env test tier2 --cov

# Stop services when done
./test-env down
```

## 🎯 Test Guidelines

1. **Unit Tests (Tier 1)**: Mock external dependencies, 1 second timeout max
2. **Integration Tests (Tier 2)**: Use real Docker services, 5 second timeout max
3. **E2E Tests (Tier 3)**: Full scenarios with real infrastructure, 10 second timeout max

## ⏱️ CRITICAL: Test Timeout Directives

**When running tests, ALWAYS enforce these timeouts:**

```bash
# Unit tests - 1 second maximum
pytest tests/unit/ --timeout=1 --timeout-method=thread

# Integration tests - 5 seconds maximum
pytest tests/integration/ --timeout=5 --timeout-method=thread

# E2E tests - 10 seconds maximum
pytest tests/e2e/ --timeout=10 --timeout-method=thread
```

### Systematic Approach to Fix Timeout Violations

1. **Identify violating tests**:
   ```bash
   # Find all tests that exceed timeout
   pytest tests/integration/ --timeout=5 -v | grep -B5 "Timeout"
   ```

2. **Common fixes for timeout violations**:
   - **Long sleeps**: Change `await asyncio.sleep(10)` → `await asyncio.sleep(0.1)`
   - **Actor cleanup**: Add proper task cancellation:
     ```python
     finally:
         if hasattr(pool, '_supervisor'):
             await pool._supervisor.stop_all_actors()
         pool._closing = True
     ```
   - **Database config**: Use fast timeouts:
     ```python
     config["health_check_interval"] = 0.1  # Not 30s!
     config["max_idle_time"] = 10.0        # Not 600s!
     ```
   - **Mock slow services**: Replace real HTTP calls with mocks
   - **Reduce iterations**: Use 2-3 iterations instead of 10+ for tests

3. **Verify fixes**:
   ```bash
   # Re-run with strict timeout to ensure fix worked
   pytest path/to/fixed_test.py --timeout=5 -v
   ```

## 🧪 Test-Driven Development (TODO-111 Pattern)

**Key Lessons from TODO-111 Implementation**:

1. **Test Before Implementation**: Write tests first to identify missing methods and architecture issues
2. **3-Tier Validation**: Each component tested at unit, integration, and E2E levels
3. **Real Infrastructure**: Use actual Docker services, not mocks, for integration tests
4. **Documentation Validation**: All examples verified with real SDK execution
5. **Comprehensive Coverage**: 100% test pass rate with meaningful scenario coverage

## 📚 Full Documentation

See **[sdk-contributors/development/](../sdk-contributors/development/)** for complete testing documentation.

## ⚡ Performance Tips

- Run tier 1 tests for fast feedback during development
- Run tier 2/3 tests before committing
- Use `-x` flag to stop on first failure: `pytest -x`
- Use `-k` to run specific tests: `pytest -k "test_order_processing"`
- Always include timeout flags to catch slow tests early
- Run with `--tb=short` for concise error output

## 📑 Test Execution Checklist for Claude Code

1. ☑️ Start test environment: `./test-env up`
2. ☑️ Run unit tests: `pytest tests/unit/ --timeout=1`
3. ☑️ Run integration tests: `pytest tests/integration/ --timeout=5`
4. ☑️ Run E2E tests: `pytest tests/e2e/ --timeout=10`
5. ☑️ Fix any timeout violations using the systematic approach above
6. ☑️ Verify all tests pass within timeout limits
