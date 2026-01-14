# Requirements Analysis: Version History UI Component

## Executive Summary

**Feature**: Visual Version History UI for Kailash Studio
**Tier**: 2B - Version Control UI (Enterprise-First Prioritization)
**Complexity**: Medium
**Risk Level**: Low
**Estimated Effort**: 8 hours
**Business Value**: $30,000 (Enterprise feature unlock)

### Overview
This component provides visual version control capabilities for workflows in Kailash Studio. It builds on the completed backend infrastructure (WorkflowVersion DataFlow model + API endpoints) to deliver an enterprise-grade version management UI with visual timeline, diff comparison, and rollback functionality.

### Backend Foundation (Completed)
- WorkflowVersion DataFlow model with 9 auto-generated nodes
- Complete CRUD API endpoints (`/api/workflow-versions`)
- Delta compression storage (snapshot-based MVP)
- Multi-tenant isolation
- 47 passing backend tests

---

## Functional Requirements Matrix

| Requirement | Description | Input | Output | Business Logic | Edge Cases | Integration Points |
|-------------|-------------|-------|--------|----------------|------------|-------------------|
| REQ-001 | Visual Timeline Display | workflow_id | Version list with metadata | Fetch versions, render chronologically | Empty history, single version, 100+ versions | ListWorkflowVersionsNode API |
| REQ-002 | Version Metadata Display | version object | Visual card with details | Format timestamp, author, message | Long commit messages, missing authors | WorkflowStore, User data |
| REQ-003 | Pagination Support | limit, offset | Paginated version list | Load versions in batches | Last page, single item, rapid scrolling | API pagination params |
| REQ-004 | Current Version Indicator | is_current flag | Visual highlight | Distinguish current from historical | Multiple current flags (data error) | WorkflowVersion.is_current |
| REQ-005 | Version Comparison Selection | Two version IDs | Diff view trigger | Store selected versions, validate selection | Same version selected twice | Component state |
| REQ-006 | Side-by-Side Diff View | version_a, version_b | Visual diff display | Compare snapshots, identify changes | Identical versions, drastically different | React Flow, diff algorithm |
| REQ-007 | Node Change Detection | Two node arrays | Added/deleted/modified nodes | Deep comparison of node data | Position-only changes, parameter changes | WorkflowNode structure |
| REQ-008 | Edge Change Detection | Two edge arrays | Added/deleted edges | Compare edge connections | Reconnected edges, handle changes | WorkflowEdge structure |
| REQ-009 | Parameter Diff Display | Node parameters | Highlighted differences | Deep object comparison | Nested objects, arrays, null values | NodeData.parameters |
| REQ-010 | Color-Coded Changes | Change type | Visual styling | Apply colors: green/red/yellow | Accessibility (colorblind modes) | CSS/Tailwind classes |
| REQ-011 | Rollback Selection | version_id | Confirmation dialog | Load version snapshot | Current version selected | ReadWorkflowVersionNode |
| REQ-012 | Rollback Preview | version snapshot | Workflow preview | Render workflow read-only | Large workflows (1000+ nodes) | WorkflowCanvas (readonly) |
| REQ-013 | Rollback Confirmation | User confirmation | New current version | Create new version from snapshot | Network failures, concurrent edits | CreateWorkflowVersionNode |
| REQ-014 | Success/Error Feedback | Operation result | Toast notification | Display status message | Timeout, partial success | NotificationToast component |

---

## Non-Functional Requirements

### Performance Requirements
```yaml
UI Responsiveness:
  - Timeline rendering: <100ms for 50 versions
  - Pagination load: <200ms per page
  - Diff calculation: <500ms for 100-node workflows
  - Preview render: <100ms (inherits WorkflowCanvas target)
  - Rollback execution: <1s total operation

Memory Constraints:
  - Timeline: Max 50 versions loaded simultaneously
  - Diff view: Two workflow snapshots + diff state
  - Total memory: <50MB for version history UI

Network Optimization:
  - API calls: Debounce pagination by 300ms
  - Diff data: Fetch only when comparison triggered
  - Snapshot data: Lazy load on preview/rollback
  - Cache: 5-minute cache for version lists
```

### Security Requirements
```yaml
Authentication:
  - JWT token validation: All API calls include auth header
  - Session management: Use existing auth store
  - Token expiry handling: Graceful re-authentication

Authorization:
  - Organization-based filtering: Only show versions for user's org
  - Permission checks: Verify user can view/rollback workflows
  - Multi-tenant isolation: Strict tenant separation in API

Data Protection:
  - Snapshot data: Treat as sensitive workflow IP
  - User information: Display only necessary author details
  - Audit logging: Log all rollback operations
```

### Accessibility Requirements
```yaml
WCAG 2.1 AA Compliance:
  - Keyboard navigation: Full timeline/diff navigation
  - Screen readers: Semantic HTML, ARIA labels
  - Color contrast: 4.5:1 minimum for text
  - Colorblind-safe: Don't rely solely on color for diff states

Usability:
  - Touch targets: 44px minimum for mobile
  - Focus indicators: Clear focus states for all interactive elements
  - Error messages: Clear, actionable error descriptions
  - Loading states: Progress indicators for async operations
```

### Scalability Requirements
```yaml
Data Volume:
  - Timeline: Support 1000+ versions per workflow
  - Pagination: 50 versions per page (configurable)
  - Diff: Handle workflows up to 1000 nodes
  - Concurrent users: 100+ simultaneous viewers

Frontend Optimization:
  - Virtual scrolling: Enabled for 100+ version timelines
  - Lazy rendering: Load diff components on-demand
  - Memoization: React.memo for version cards
  - Debouncing: User interactions debounced 300ms
```

---

## User Journey Mapping

### Primary Persona: Enterprise Developer
**Goal**: Track workflow changes and rollback if needed

#### Journey 1: View Version History
```
1. Open workflow in WorkflowCanvas
   → Click "Version History" button in toolbar
   → UI: Version history panel slides in from right

2. View timeline of versions
   → See: Version cards with number, author, timestamp, message
   → Visual: Current version highlighted in blue
   → Success: All versions loaded <200ms

3. Scroll through older versions
   → Action: Scroll down timeline
   → Pagination: Auto-load next 50 versions
   → Success: Smooth scrolling, no janky rendering

Success Criteria:
✓ Timeline renders in <100ms
✓ All versions visible with clear metadata
✓ Current version clearly distinguished
✓ Pagination seamless (no loading flickers)

Failure Points:
✗ Empty history (no versions exist)
   → Mitigation: "No version history yet" empty state
✗ API timeout/error
   → Mitigation: Error message + retry button
✗ Slow pagination
   → Mitigation: Loading skeleton, virtual scroll
```

#### Journey 2: Compare Two Versions
```
1. Select first version from timeline
   → Click: Version card checkbox
   → UI: Card highlighted, "Compare" button appears

2. Select second version
   → Click: Different version card checkbox
   → UI: Both cards highlighted
   → Validation: Can't select more than 2

3. Open comparison view
   → Click: "Compare Versions" button
   → Loading: Calculate diff (show spinner)
   → UI: Split-pane diff view opens

4. Review changes
   → See: Side-by-side workflow views
   → Highlights:
     - Green boxes = Added nodes
     - Red boxes = Deleted nodes
     - Yellow boxes = Modified nodes
   → Detail: Click node to see parameter diff

Success Criteria:
✓ Diff calculated in <500ms for 100 nodes
✓ All changes clearly visualized
✓ Parameter diffs drill-down functional
✓ Color coding accessible (icons too)

Failure Points:
✗ Identical versions selected
   → Mitigation: Disable compare, show "No changes"
✗ Large workflow diff (1000+ nodes)
   → Mitigation: Performance mode, summarized diff
✗ Complex parameter changes
   → Mitigation: Expandable JSON diff viewer
```

#### Journey 3: Rollback to Previous Version
```
1. Select version to restore
   → Click: "Rollback" button on version card
   → UI: Confirmation dialog opens

2. Preview workflow
   → See: Read-only preview of workflow at that version
   → Review: Nodes, edges, parameters
   → Decision: Confirm or cancel

3. Confirm rollback
   → Click: "Confirm Rollback" button
   → Warning: "This will create a new version with this snapshot"
   → Action: User confirms

4. Execute rollback
   → API: CreateWorkflowVersionNode with snapshot
   → Loading: "Rolling back..." progress indicator
   → Success: "Rollback complete" toast notification
   → UI: Timeline updates, new version appears as current

Success Criteria:
✓ Preview renders in <100ms
✓ Rollback completes in <1s
✓ New version marked as current
✓ Success notification shown

Failure Points:
✗ Network failure during rollback
   → Mitigation: Retry logic + error message
✗ Concurrent modification (someone else editing)
   → Mitigation: Warning dialog about conflicts
✗ Permission denied
   → Mitigation: Clear error message, hide rollback button
```

### Secondary Persona: Business Analyst (View-Only)
**Goal**: Understand workflow evolution for compliance audit

#### Journey 4: Audit Trail Review
```
1. Open version history for audit
   → See: Complete timeline of all changes
   → Filter: By date range, author

2. Review commit messages
   → Scan: Understand why changes were made
   → Export: Copy version history for documentation

3. Generate change report
   → Select: Date range
   → Export: CSV of version metadata

Success Criteria:
✓ Filter by date/author functional
✓ Export generates complete report
✓ All commit messages visible

Failure Points:
✗ Missing commit messages
   → Mitigation: Show "(No message)" placeholder
✗ Too many versions to review
   → Mitigation: Search/filter functionality
```

---

## Component Architecture

### Component Hierarchy
```
VersionHistoryPanel (Container)
├── VersionHistoryHeader
│   ├── Title: "Version History"
│   ├── CloseButton
│   └── FilterControls (author, date range)
│
├── VersionTimeline (List)
│   ├── VirtualScroller (for 100+ versions)
│   ├── VersionCard[] (repeating)
│   │   ├── VersionNumber
│   │   ├── AuthorAvatar
│   │   ├── Timestamp
│   │   ├── CommitMessage
│   │   ├── CurrentBadge (conditional)
│   │   ├── SelectCheckbox (for compare)
│   │   └── ActionButtons (rollback, view)
│   └── PaginationLoader (bottom)
│
├── CompareToolbar (conditional: 2 selected)
│   ├── SelectedVersions display
│   ├── CompareButton
│   └── ClearSelectionButton
│
├── DiffViewer (modal/drawer)
│   ├── DiffViewerHeader
│   │   ├── VersionLabels (A vs B)
│   │   └── CloseButton
│   ├── SideBySidePanels
│   │   ├── WorkflowPreview (version A)
│   │   │   └── React Flow (read-only, highlighted)
│   │   └── WorkflowPreview (version B)
│   │       └── React Flow (read-only, highlighted)
│   ├── ChangeSummary
│   │   ├── NodesAdded count (green)
│   │   ├── NodesDeleted count (red)
│   │   ├── NodesModified count (yellow)
│   │   └── EdgesChanged count
│   └── ParameterDiffPanel (expandable)
│       └── JSONDiff viewer
│
└── RollbackDialog (modal)
    ├── WarningMessage
    ├── VersionPreview
    │   └── WorkflowCanvas (read-only)
    ├── CommitMessageInput (for new version)
    └── ActionButtons
        ├── CancelButton
        └── ConfirmButton
```

### Props and State Management

#### VersionHistoryPanel Props
```typescript
interface VersionHistoryPanelProps {
  workflowId: string;              // Current workflow being viewed
  isOpen: boolean;                 // Panel visibility
  onClose: () => void;             // Close handler
  onRollbackSuccess?: (newVersionId: string) => void;  // Callback
}
```

#### Component State (Zustand Store Extension)
```typescript
interface VersionHistoryStore {
  // Version data
  versions: WorkflowVersion[];              // Loaded versions
  totalCount: number;                        // Total versions available
  currentPage: number;                       // Pagination state
  isLoading: boolean;                        // Loading indicator
  error: string | null;                      // Error state

  // Selection state
  selectedVersionIds: string[];              // For comparison (max 2)

  // Diff state
  diffViewOpen: boolean;                     // Diff modal visibility
  diffData: {
    versionA: WorkflowVersion;
    versionB: WorkflowVersion;
    changes: DiffResult;
  } | null;

  // Rollback state
  rollbackDialogOpen: boolean;
  rollbackVersion: WorkflowVersion | null;
  rollbackInProgress: boolean;

  // Actions
  loadVersions: (workflowId: string, page: number) => Promise<void>;
  selectVersion: (versionId: string) => void;
  deselectVersion: (versionId: string) => void;
  clearSelection: () => void;
  openDiffView: (versionAId: string, versionBId: string) => Promise<void>;
  closeDiffView: () => void;
  initiateRollback: (versionId: string) => void;
  executeRollback: (commitMessage: string) => Promise<void>;
  cancelRollback: () => void;
}

// New type definitions
interface DiffResult {
  nodesAdded: WorkflowNode[];
  nodesDeleted: WorkflowNode[];
  nodesModified: Array<{
    nodeId: string;
    before: WorkflowNode;
    after: WorkflowNode;
    parameterChanges: ParameterDiff[];
  }>;
  edgesAdded: WorkflowEdge[];
  edgesDeleted: WorkflowEdge[];
  summary: {
    totalNodesChanged: number;
    totalEdgesChanged: number;
  };
}

interface ParameterDiff {
  path: string;                    // e.g., "parameters.api_key"
  before: any;
  after: any;
  changeType: 'added' | 'deleted' | 'modified';
}
```

### Integration Points

#### 1. WorkflowCanvas Integration
```typescript
// Add version history button to WorkflowCanvas toolbar
<WorkflowCanvas>
  <Toolbar>
    <Button onClick={openVersionHistory}>
      <HistoryIcon /> Version History
    </Button>
  </Toolbar>
</WorkflowCanvas>

// Pass current workflow ID to version history panel
<VersionHistoryPanel
  workflowId={currentWorkflow.id}
  isOpen={versionHistoryOpen}
  onClose={() => setVersionHistoryOpen(false)}
/>
```

#### 2. Backend API Integration
```typescript
// API Service Layer
class VersionHistoryAPI {
  // List versions with pagination
  async listVersions(
    workflowId: string,
    params: { limit?: number; offset?: number }
  ): Promise<{ versions: WorkflowVersion[]; pagination: Pagination }> {
    return apiClient.get(`/api/workflow-versions`, {
      params: { workflow_id: workflowId, ...params }
    });
  }

  // Get single version
  async getVersion(versionId: string): Promise<WorkflowVersion> {
    return apiClient.get(`/api/workflow-versions/${versionId}`);
  }

  // Create new version (for rollback)
  async createVersion(data: CreateVersionRequest): Promise<WorkflowVersion> {
    return apiClient.post(`/api/workflow-versions`, data);
  }
}
```

#### 3. Zustand Store Integration
```typescript
// Extend existing workflow store
export const useWorkflowStore = create<WorkflowStore>()(
  devtools(
    immer((set, get) => ({
      // Existing workflow state...

      // Add version history state
      versionHistory: {
        versions: [],
        selectedVersionIds: [],
        // ... rest of version history state
      },

      // Add version history actions
      loadVersionHistory: async (workflowId: string) => {
        set(state => { state.versionHistory.isLoading = true; });
        try {
          const result = await versionHistoryAPI.listVersions(workflowId);
          set(state => {
            state.versionHistory.versions = result.versions;
            state.versionHistory.totalCount = result.pagination.total;
            state.versionHistory.isLoading = false;
          });
        } catch (error) {
          set(state => {
            state.versionHistory.error = error.message;
            state.versionHistory.isLoading = false;
          });
        }
      },
    }))
  )
);
```

#### 4. React Flow Integration (Diff View)
```typescript
// Render diff view with React Flow
<DiffViewer>
  <SplitPane>
    {/* Left: Version A */}
    <ReactFlowProvider>
      <ReactFlow
        nodes={highlightChanges(versionA.nodes, diffData, 'before')}
        edges={versionA.edges}
        nodeTypes={diffNodeTypes}  // Custom colored nodes
        fitView
        interactive={false}        // Read-only
      />
    </ReactFlowProvider>

    {/* Right: Version B */}
    <ReactFlowProvider>
      <ReactFlow
        nodes={highlightChanges(versionB.nodes, diffData, 'after')}
        edges={versionB.edges}
        nodeTypes={diffNodeTypes}
        fitView
        interactive={false}
      />
    </ReactFlowProvider>
  </SplitPane>
</DiffViewer>

// Custom node styling for diff
const diffNodeTypes = {
  added: (props) => <Node {...props} style={{ border: '2px solid green' }} />,
  deleted: (props) => <Node {...props} style={{ border: '2px solid red' }} />,
  modified: (props) => <Node {...props} style={{ border: '2px solid yellow' }} />,
  unchanged: (props) => <Node {...props} style={{ opacity: 0.5 }} />,
};
```

---

## Technical Specifications

### API Contracts

#### GET /api/workflow-versions
```typescript
Request:
  Query Parameters:
    - workflow_id: string (required)
    - limit: number (default: 50, max: 100)
    - offset: number (default: 0)
    - author_id: string (optional, filter)
    - is_current: boolean (optional, filter)

Response: 200 OK
  {
    "success": true,
    "data": {
      "versions": [
        {
          "id": "wfv_abc123",
          "workflow_id": "wf_xyz789",
          "version_number": 5,
          "commit_message": "Updated approval logic",
          "snapshot_data": { "nodes": [...], "edges": [...] },
          "author_id": "user_123",
          "organization_id": "org_456",
          "is_current": true,
          "size_bytes": 4521,
          "metadata": {},
          "created_at": "2025-10-05T14:30:00Z",
          "updated_at": "2025-10-05T14:30:00Z"
        }
      ],
      "pagination": {
        "total": 127,
        "limit": 50,
        "offset": 0,
        "has_more": true
      }
    }
  }

Error Response: 400/500
  {
    "error": "Error message",
    "code": "ERROR_CODE"
  }
```

#### POST /api/workflow-versions (Rollback)
```typescript
Request:
  Body:
    {
      "workflow_id": "wf_xyz789",
      "commit_message": "Rollback to version 3",
      "snapshot_data": { /* copied from previous version */ },
      "author_id": "user_123",
      "organization_id": "org_456",
      "is_current": true,
      "metadata": {
        "rollback_from_version": 5,
        "rollback_to_version": 3
      }
    }

Response: 201 Created
  {
    "success": true,
    "data": {
      "id": "wfv_new456",
      "version_number": 6,
      // ... complete version object
    },
    "message": "Version created successfully"
  }
```

### Data Structures

#### WorkflowVersion (Frontend Type)
```typescript
interface WorkflowVersion {
  id: string;
  workflow_id: string;
  version_number: number;
  commit_message: string;
  snapshot_data: {
    nodes: WorkflowNode[];
    edges: WorkflowEdge[];
    metadata: WorkflowMetadata;
  };
  author_id: string;
  organization_id: string;
  is_current: boolean;
  size_bytes: number;
  metadata: {
    rollback_from_version?: number;
    rollback_to_version?: number;
    [key: string]: any;
  };
  created_at: string;  // ISO 8601 timestamp
  updated_at: string;
}
```

#### Diff Algorithm
```typescript
// Utility function for computing workflow diff
function computeWorkflowDiff(
  versionA: WorkflowVersion,
  versionB: WorkflowVersion
): DiffResult {
  const nodesA = new Map(versionA.snapshot_data.nodes.map(n => [n.id, n]));
  const nodesB = new Map(versionB.snapshot_data.nodes.map(n => [n.id, n]));

  const nodesAdded: WorkflowNode[] = [];
  const nodesDeleted: WorkflowNode[] = [];
  const nodesModified: Array<{ /* ... */ }> = [];

  // Detect added nodes (in B, not in A)
  for (const [id, node] of nodesB) {
    if (!nodesA.has(id)) {
      nodesAdded.push(node);
    }
  }

  // Detect deleted nodes (in A, not in B)
  for (const [id, node] of nodesA) {
    if (!nodesB.has(id)) {
      nodesDeleted.push(node);
    }
  }

  // Detect modified nodes (in both, but different)
  for (const [id, nodeB] of nodesB) {
    const nodeA = nodesA.get(id);
    if (nodeA && !deepEqual(nodeA, nodeB)) {
      const paramChanges = computeParameterDiff(nodeA.data, nodeB.data);
      nodesModified.push({
        nodeId: id,
        before: nodeA,
        after: nodeB,
        parameterChanges: paramChanges,
      });
    }
  }

  // Compute edge diff (similar logic)
  const edgesAdded = /* ... */;
  const edgesDeleted = /* ... */;

  return {
    nodesAdded,
    nodesDeleted,
    nodesModified,
    edgesAdded,
    edgesDeleted,
    summary: {
      totalNodesChanged: nodesAdded.length + nodesDeleted.length + nodesModified.length,
      totalEdgesChanged: edgesAdded.length + edgesDeleted.length,
    },
  };
}

// Deep object comparison for parameters
function computeParameterDiff(dataA: NodeData, dataB: NodeData): ParameterDiff[] {
  // Use library like 'deep-object-diff' or custom recursive comparison
  const diffs: ParameterDiff[] = [];

  // Compare parameters recursively
  const allKeys = new Set([
    ...Object.keys(dataA.parameters || {}),
    ...Object.keys(dataB.parameters || {}),
  ]);

  for (const key of allKeys) {
    const valueA = dataA.parameters?.[key];
    const valueB = dataB.parameters?.[key];

    if (valueA === undefined && valueB !== undefined) {
      diffs.push({ path: key, before: null, after: valueB, changeType: 'added' });
    } else if (valueA !== undefined && valueB === undefined) {
      diffs.push({ path: key, before: valueA, after: null, changeType: 'deleted' });
    } else if (!deepEqual(valueA, valueB)) {
      diffs.push({ path: key, before: valueA, after: valueB, changeType: 'modified' });
    }
  }

  return diffs;
}
```

### Performance Considerations

#### 1. Virtual Scrolling for Timeline
```typescript
import { FixedSizeList as List } from 'react-window';

function VersionTimeline({ versions }: { versions: WorkflowVersion[] }) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <VersionCard version={versions[index]} />
    </div>
  );

  return (
    <List
      height={600}               // Viewport height
      itemCount={versions.length}
      itemSize={120}             // Each version card height
      width="100%"
    >
      {Row}
    </List>
  );
}
```

#### 2. Debounced Pagination
```typescript
import { useDebouncedCallback } from 'use-debounce';

function useVersionPagination(workflowId: string) {
  const loadMore = useDebouncedCallback(
    async (page: number) => {
      await versionHistoryAPI.listVersions(workflowId, {
        offset: page * 50,
        limit: 50
      });
    },
    300  // Debounce 300ms
  );

  return { loadMore };
}
```

#### 3. Memoized Diff Calculation
```typescript
import { useMemo } from 'react';

function DiffViewer({ versionA, versionB }: DiffViewerProps) {
  // Memoize expensive diff calculation
  const diffResult = useMemo(
    () => computeWorkflowDiff(versionA, versionB),
    [versionA.id, versionB.id]  // Only recalculate if versions change
  );

  return <DiffDisplay diff={diffResult} />;
}
```

#### 4. Lazy Loading Snapshot Data
```typescript
// Only load full snapshot when preview/rollback triggered
async function loadVersionSnapshot(versionId: string): Promise<WorkflowDefinition> {
  // Timeline only loads metadata (no snapshot_data)
  const version = await versionHistoryAPI.getVersion(versionId);
  return version.snapshot_data;
}
```

---

## Implementation Plan

### Phase 1: Foundation Components (3h)
**Goal**: Build core UI structure with timeline display

```yaml
Tasks:
  1. Create VersionHistoryPanel component shell
     - Drawer/modal container
     - Header with close button
     - Empty state placeholder

  2. Implement VersionTimeline with basic rendering
     - VersionCard component
     - Display version metadata (number, author, timestamp, message)
     - Current version badge/highlight

  3. Set up Zustand store extension
     - Version history state slice
     - Basic actions (load, select, clear)

  4. API service integration
     - VersionHistoryAPI class
     - GET /api/workflow-versions integration

  5. Basic pagination
     - Load first 50 versions
     - "Load More" button

Deliverables:
  - VersionHistoryPanel.tsx
  - VersionCard.tsx
  - VersionTimeline.tsx
  - store/versionHistory.ts
  - services/VersionHistoryAPI.ts

Tests:
  - VersionHistoryPanel renders correctly
  - VersionCard displays metadata
  - Timeline loads and displays versions
  - Pagination loads next page
  - Current version is highlighted
```

### Phase 2: Comparison & Diff View (3h)
**Goal**: Enable version comparison with visual diff

```yaml
Tasks:
  1. Selection mechanism
     - Checkbox on version cards
     - Select up to 2 versions
     - "Compare" button appears when 2 selected

  2. Diff calculation
     - computeWorkflowDiff utility function
     - Node comparison (added/deleted/modified)
     - Edge comparison
     - Parameter diff for modified nodes

  3. DiffViewer component
     - Side-by-side layout
     - React Flow integration for visual diff
     - Custom node styling (green/red/yellow)
     - Change summary panel

  4. ParameterDiffPanel
     - Expandable JSON diff viewer
     - Highlight added/deleted/modified fields
     - Nested object support

Deliverables:
  - CompareToolbar.tsx
  - DiffViewer.tsx
  - SideBySidePanels.tsx
  - ParameterDiffPanel.tsx
  - utils/diffAlgorithm.ts

Tests:
  - Can select 2 versions
  - Cannot select more than 2
  - Diff calculates correctly for nodes
  - Diff calculates correctly for edges
  - Parameter diff shows nested changes
  - Visual highlighting correct (colors + icons)
```

### Phase 3: Rollback Functionality (2h)
**Goal**: Implement version rollback with preview and confirmation

```yaml
Tasks:
  1. Rollback initiation
     - "Rollback" button on version cards
     - RollbackDialog modal

  2. Preview functionality
     - Load version snapshot
     - Render workflow in read-only WorkflowCanvas
     - Show version metadata

  3. Confirmation flow
     - Warning message about creating new version
     - Commit message input
     - Confirm/Cancel buttons

  4. Rollback execution
     - Call CreateWorkflowVersionNode API
     - Handle success/failure
     - Update timeline with new version
     - Show success toast notification

Deliverables:
  - RollbackDialog.tsx
  - RollbackPreview.tsx
  - Rollback action in store
  - Success/error handling

Tests:
  - Rollback dialog opens
  - Preview renders correctly
  - Can cancel rollback
  - Rollback creates new version
  - Success notification shown
  - Timeline updates after rollback
  - Error handling for network failures
```

---

## Test Requirements (TDD Approach)

### Tier 1: Unit Tests (Component Behavior)

#### VersionHistoryPanel.unit.test.tsx
```typescript
describe('VersionHistoryPanel', () => {
  it('renders when open', () => {
    render(<VersionHistoryPanel isOpen={true} workflowId="wf_123" onClose={jest.fn()} />);
    expect(screen.getByText('Version History')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(<VersionHistoryPanel isOpen={false} workflowId="wf_123" onClose={jest.fn()} />);
    expect(screen.queryByText('Version History')).not.toBeInTheDocument();
  });

  it('calls onClose when close button clicked', () => {
    const onClose = jest.fn();
    render(<VersionHistoryPanel isOpen={true} workflowId="wf_123" onClose={onClose} />);
    fireEvent.click(screen.getByLabelText('Close'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows loading state', () => {
    // Mock loading state in store
    render(<VersionHistoryPanel isOpen={true} workflowId="wf_123" onClose={jest.fn()} />);
    expect(screen.getByText('Loading versions...')).toBeInTheDocument();
  });

  it('shows empty state when no versions', () => {
    // Mock empty versions array
    render(<VersionHistoryPanel isOpen={true} workflowId="wf_123" onClose={jest.fn()} />);
    expect(screen.getByText('No version history yet')).toBeInTheDocument();
  });
});
```

#### VersionCard.unit.test.tsx
```typescript
describe('VersionCard', () => {
  const mockVersion: WorkflowVersion = {
    id: 'wfv_123',
    workflow_id: 'wf_456',
    version_number: 5,
    commit_message: 'Updated approval logic',
    author_id: 'user_789',
    is_current: true,
    created_at: '2025-10-05T14:30:00Z',
    // ... rest of fields
  };

  it('displays version number', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByText('Version 5')).toBeInTheDocument();
  });

  it('displays commit message', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByText('Updated approval logic')).toBeInTheDocument();
  });

  it('displays timestamp', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByText(/Oct 5, 2025/)).toBeInTheDocument();
  });

  it('shows current badge for current version', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByText('Current')).toBeInTheDocument();
  });

  it('does not show current badge for historical version', () => {
    const historicalVersion = { ...mockVersion, is_current: false };
    render(<VersionCard version={historicalVersion} />);
    expect(screen.queryByText('Current')).not.toBeInTheDocument();
  });

  it('has rollback button', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByRole('button', { name: /rollback/i })).toBeInTheDocument();
  });

  it('has select checkbox', () => {
    render(<VersionCard version={mockVersion} />);
    expect(screen.getByRole('checkbox')).toBeInTheDocument();
  });
});
```

#### DiffViewer.unit.test.tsx
```typescript
describe('DiffViewer', () => {
  const versionA: WorkflowVersion = { /* version 3 */ };
  const versionB: WorkflowVersion = { /* version 5 */ };

  it('renders side-by-side panels', () => {
    render(<DiffViewer versionA={versionA} versionB={versionB} />);
    expect(screen.getByText('Version 3')).toBeInTheDocument();
    expect(screen.getByText('Version 5')).toBeInTheDocument();
  });

  it('displays change summary', () => {
    render(<DiffViewer versionA={versionA} versionB={versionB} />);
    expect(screen.getByText(/2 nodes added/i)).toBeInTheDocument();
    expect(screen.getByText(/1 node deleted/i)).toBeInTheDocument();
    expect(screen.getByText(/3 nodes modified/i)).toBeInTheDocument();
  });

  it('highlights added nodes in green', () => {
    render(<DiffViewer versionA={versionA} versionB={versionB} />);
    const addedNode = screen.getByTestId('node-added-123');
    expect(addedNode).toHaveStyle({ borderColor: 'green' });
  });

  it('highlights deleted nodes in red', () => {
    render(<DiffViewer versionA={versionA} versionB={versionB} />);
    const deletedNode = screen.getByTestId('node-deleted-456');
    expect(deletedNode).toHaveStyle({ borderColor: 'red' });
  });

  it('shows parameter diff when node clicked', () => {
    render(<DiffViewer versionA={versionA} versionB={versionB} />);
    fireEvent.click(screen.getByTestId('node-modified-789'));
    expect(screen.getByText('Parameter Changes')).toBeInTheDocument();
  });
});
```

### Tier 2: Integration Tests (Real Backend)

#### VersionHistory.integration.test.tsx
```typescript
describe('Version History Integration', () => {
  let testWorkflowId: string;
  let testVersionIds: string[];

  beforeAll(async () => {
    // Create test workflow and versions using real backend
    testWorkflowId = await createTestWorkflow();
    testVersionIds = await createTestVersions(testWorkflowId, 5);
  });

  afterAll(async () => {
    // Clean up test data
    await deleteTestWorkflow(testWorkflowId);
  });

  it('loads versions from real API', async () => {
    render(<VersionHistoryPanel isOpen={true} workflowId={testWorkflowId} onClose={jest.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Version 1')).toBeInTheDocument();
      expect(screen.getByText('Version 5')).toBeInTheDocument();
    });
  });

  it('paginates through large version history', async () => {
    // Create 100 versions
    await createTestVersions(testWorkflowId, 100);

    render(<VersionHistoryPanel isOpen={true} workflowId={testWorkflowId} onClose={jest.fn()} />);

    // Should load first 50
    await waitFor(() => {
      expect(screen.getByText('Version 1')).toBeInTheDocument();
    });

    // Load more
    fireEvent.click(screen.getByText('Load More'));

    await waitFor(() => {
      expect(screen.getByText('Version 51')).toBeInTheDocument();
    });
  });

  it('performs rollback successfully', async () => {
    render(<VersionHistoryPanel isOpen={true} workflowId={testWorkflowId} onClose={jest.fn()} />);

    // Click rollback on version 3
    const rollbackButton = screen.getByTestId('rollback-version-3');
    fireEvent.click(rollbackButton);

    // Confirm rollback
    fireEvent.change(screen.getByLabelText('Commit message'), {
      target: { value: 'Rollback to version 3' },
    });
    fireEvent.click(screen.getByText('Confirm Rollback'));

    // Wait for success
    await waitFor(() => {
      expect(screen.getByText('Rollback complete')).toBeInTheDocument();
    });

    // Verify new version created
    await waitFor(() => {
      expect(screen.getByText('Version 6')).toBeInTheDocument();
      const newVersion = screen.getByTestId('version-6');
      expect(within(newVersion).getByText('Current')).toBeInTheDocument();
    });
  });

  it('handles API errors gracefully', async () => {
    // Simulate network error
    mockApiError(500);

    render(<VersionHistoryPanel isOpen={true} workflowId={testWorkflowId} onClose={jest.fn()} />);

    await waitFor(() => {
      expect(screen.getByText(/failed to load versions/i)).toBeInTheDocument();
    });

    // Retry button works
    fireEvent.click(screen.getByText('Retry'));
    await waitFor(() => {
      expect(screen.getByText('Version 1')).toBeInTheDocument();
    });
  });
});
```

### Tier 3: Performance Tests

#### VersionHistory.performance.test.tsx
```typescript
describe('Version History Performance', () => {
  it('renders 50 versions in <100ms', async () => {
    const versions = generateMockVersions(50);

    const startTime = performance.now();
    render(<VersionTimeline versions={versions} />);
    const renderTime = performance.now() - startTime;

    expect(renderTime).toBeLessThan(100);
  });

  it('calculates diff for 100-node workflow in <500ms', () => {
    const versionA = generateMockVersion(100);  // 100 nodes
    const versionB = generateMockVersion(100);

    const startTime = performance.now();
    const diff = computeWorkflowDiff(versionA, versionB);
    const calcTime = performance.now() - startTime;

    expect(calcTime).toBeLessThan(500);
  });

  it('handles 1000+ version timeline with virtual scrolling', () => {
    const versions = generateMockVersions(1000);

    const { container } = render(<VersionTimeline versions={versions} />);

    // Only rendered items should be in DOM (not all 1000)
    const renderedCards = container.querySelectorAll('.version-card');
    expect(renderedCards.length).toBeLessThan(50);  // Virtual window
  });
});
```

---

## Acceptance Criteria

### Functional Acceptance
- [ ] Version timeline displays all versions with complete metadata
- [ ] Current version is clearly distinguished (badge + highlight)
- [ ] Pagination works smoothly with no UI jank
- [ ] Can select exactly 2 versions for comparison
- [ ] Diff view shows all added/deleted/modified nodes
- [ ] Parameter diff displays nested changes correctly
- [ ] Rollback preview renders workflow accurately
- [ ] Rollback creates new version and updates timeline
- [ ] Success/error notifications appear for all operations

### Performance Acceptance
- [ ] Timeline renders 50 versions in <100ms
- [ ] Pagination loads next page in <200ms
- [ ] Diff calculation completes in <500ms for 100-node workflows
- [ ] Rollback preview renders in <100ms
- [ ] Rollback operation completes in <1s

### Accessibility Acceptance
- [ ] Full keyboard navigation (Tab, Enter, Escape)
- [ ] Screen reader announces all version info
- [ ] Color contrast meets WCAG AA (4.5:1)
- [ ] Diff uses icons + color (not color alone)
- [ ] Focus indicators visible on all interactive elements

### Security Acceptance
- [ ] All API calls include JWT authentication
- [ ] Only versions from user's organization displayed
- [ ] Permission checks prevent unauthorized rollbacks
- [ ] Snapshot data treated as sensitive IP

### Integration Acceptance
- [ ] Works seamlessly with WorkflowCanvas
- [ ] Integrates with existing Zustand store
- [ ] Uses NotificationToast for feedback
- [ ] Respects organization-based multi-tenancy
- [ ] Backend API endpoints functioning correctly

---

## Risks and Mitigations

### Technical Risks

#### Risk 1: Diff Performance for Large Workflows
**Probability**: Medium
**Impact**: High
**Symptoms**: Slow diff calculation (>1s) for workflows with 500+ nodes

**Mitigation Strategies**:
1. **Web Workers**: Offload diff calculation to background thread
   ```typescript
   const diffWorker = new Worker('diffCalculator.worker.ts');
   diffWorker.postMessage({ versionA, versionB });
   diffWorker.onmessage = (e) => setDiffResult(e.data);
   ```

2. **Progressive Rendering**: Show summary first, details on-demand
   ```typescript
   // Show summary immediately
   <ChangeSummary diff={diff.summary} />

   // Load detailed diff lazily
   <LazyLoad>
     <DetailedDiffView diff={diff.details} />
   </LazyLoad>
   ```

3. **Diff Caching**: Cache diff results for 5 minutes
   ```typescript
   const cacheKey = `diff:${versionA.id}:${versionB.id}`;
   const cachedDiff = diffCache.get(cacheKey);
   if (cachedDiff) return cachedDiff;
   ```

#### Risk 2: Memory Consumption for Version Timeline
**Probability**: Medium
**Impact**: Medium
**Symptoms**: Browser slowdown when 100+ versions loaded

**Mitigation Strategies**:
1. **Virtual Scrolling**: Already planned (react-window)
2. **Pagination**: Limit to 50 versions per page
3. **Lazy Loading**: Only load snapshot_data when needed
   ```typescript
   // Timeline loads minimal metadata
   { id, version_number, commit_message, author_id, created_at }

   // Snapshot loaded only for preview/diff
   await loadVersionSnapshot(versionId);
   ```

#### Risk 3: Race Conditions in Rollback
**Probability**: Low
**Impact**: High
**Symptoms**: Concurrent rollbacks corrupt workflow state

**Mitigation Strategies**:
1. **Optimistic Locking**: Check version number before rollback
   ```typescript
   if (currentWorkflow.version !== expectedVersion) {
     throw new Error('Workflow was modified. Please refresh.');
   }
   ```

2. **Rollback Queue**: Serialize rollback operations
   ```typescript
   const rollbackQueue = new PQueue({ concurrency: 1 });
   await rollbackQueue.add(() => executeRollback(versionId));
   ```

3. **User Warning**: Alert if concurrent modifications detected
   ```typescript
   if (hasConflict) {
     showWarning('Another user modified this workflow. Continue anyway?');
   }
   ```

### Business Risks

#### Risk 4: User Confusion About Version Semantics
**Probability**: Medium
**Impact**: Medium
**Symptoms**: Users think rollback will "undo" when it creates new version

**Mitigation Strategies**:
1. **Clear Messaging**: Explain rollback creates new version
   ```
   "Rollback will create a NEW version (#6) using the snapshot from version #3.
   This does not delete versions 4 or 5."
   ```

2. **Visual Indicator**: Show version tree/timeline after rollback
   ```
   v1 → v2 → v3 → v4 → v5 (current)
                ↓ (rollback)
   v1 → v2 → v3 → v4 → v5 → v6 (new current, copy of v3)
   ```

3. **Confirmation Dialog**: Require explicit understanding
   ```
   ☑ I understand rollback creates a new version
   ☑ I understand this does not delete existing versions
   ```

#### Risk 5: Excessive Version Creation
**Probability**: High
**Impact**: Low
**Symptoms**: Users create version on every small change, filling storage

**Mitigation Strategies**:
1. **Auto-Save Strategy**: Group small changes into periodic snapshots
2. **Version Limits**: Warn when 100+ versions exist
3. **Cleanup Policy**: Archive versions older than 90 days (enterprise setting)

### UX Risks

#### Risk 6: Overwhelming UI for New Users
**Probability**: Medium
**Impact**: Medium
**Symptoms**: Users confused by too many version management options

**Mitigation Strategies**:
1. **Progressive Disclosure**: Hide advanced features initially
   ```typescript
   <VersionHistoryPanel>
     <BasicView />  {/* Default: simple timeline */}
     <AdvancedButton onClick={showAdvanced}>
       Advanced Options
     </AdvancedButton>
   </VersionHistoryPanel>
   ```

2. **Tooltips and Help**: Inline guidance for all features
3. **Onboarding Tour**: First-time user guide for version history

---

## Dependencies

### Technical Dependencies
- **Completed**:
  - ✅ WorkflowVersion DataFlow model
  - ✅ Backend API endpoints (/api/workflow-versions)
  - ✅ WorkflowCanvas component
  - ✅ Zustand store infrastructure
  - ✅ React Flow library

- **Required**:
  - `react-window` (virtual scrolling)
  - `deep-object-diff` (parameter comparison)
  - `date-fns` (timestamp formatting)
  - `lucide-react` (icons for add/delete/modify)

### Development Dependencies
- **Frontend Team**: 1 React developer (8h)
- **Design Assets**: Version history icons, diff color palette
- **API Access**: Backend running locally or staging environment

### External Dependencies
- **Backend Availability**: `/api/workflow-versions` endpoints accessible
- **Test Data**: Ability to create test workflows and versions
- **Authentication**: JWT token from existing auth system

---

## Success Metrics

### Immediate Metrics (Post-Implementation)
- **Test Coverage**: >90% for all version history components
- **Performance**: All targets met (<100ms UI, <500ms diff)
- **Accessibility**: WCAG AA compliance verified
- **Zero Critical Bugs**: No P0/P1 bugs in staging

### 30-Day Metrics
- **Adoption**: 80% of active users use version history at least once
- **Rollback Usage**: 10+ rollback operations per day
- **Comparison Usage**: 50+ diff views per day
- **User Feedback**: >4.0/5 satisfaction rating

### 90-Day Metrics
- **Enterprise Value**: Version history cited in 30% of enterprise demos
- **Support Tickets**: <5 version history-related tickets per week
- **Performance**: Zero performance-related complaints
- **Feature Requests**: Prioritize based on user feedback

---

## Related Documents

### Architecture Decision Records
- [ADR-0050: Kailash Studio Visual Workflow Platform](/Users/esperie/repos/projects/kailash_python_sdk/docs/adr/0050-kailash-studio-visual-workflow-platform.md)
- ADR-0051: Version History UI Component (to be created after requirements approval)

### Backend Documentation
- WorkflowVersion Model: `/apps/kailash-studio/backend/src/kailash_studio/models.py:387-427`
- Version Control API: `/apps/kailash-studio/backend/src/kailash_studio/api/workflow_versions.py`
- Backend Tests: `/apps/kailash-studio/backend/tests/test_mcp_integration.py` (47 tests)

### Frontend Documentation
- WorkflowCanvas Component: `/apps/kailash-studio/frontend/src/components/WorkflowCanvas.tsx`
- Workflow Store: `/apps/kailash-studio/frontend/src/store/workflow.ts`
- Type Definitions: `/apps/kailash-studio/frontend/src/types/index.ts`

### Testing Standards
- TESTING_STANDARDS.md (referenced in unit tests)
- Test Fixtures: `/apps/kailash-studio/frontend/src/__tests__/test-fixtures/`

---

## Appendix: Wire frames and Visual Mockups

### Version History Timeline
```
┌─────────────────────────────────────────────────┐
│ Version History                            [×]  │
├─────────────────────────────────────────────────┤
│ Filter: [All Authors ▾] [Date Range ▾]         │
├─────────────────────────────────────────────────┤
│                                                 │
│ ┌─────────────────────────────────────────────┐│
│ │ [☑] Version 5 · Current                     ││
│ │ John Doe · Oct 5, 2025 2:30 PM             ││
│ │ "Updated approval logic for expenses"       ││
│ │ [Rollback] [View]                           ││
│ └─────────────────────────────────────────────┘│
│                                                 │
│ ┌─────────────────────────────────────────────┐│
│ │ [☑] Version 4                               ││
│ │ Jane Smith · Oct 4, 2025 4:15 PM           ││
│ │ "Fixed validation error"                    ││
│ │ [Rollback] [View]                           ││
│ └─────────────────────────────────────────────┘│
│                                                 │
│ ┌─────────────────────────────────────────────┐│
│ │ [ ] Version 3                               ││
│ │ John Doe · Oct 3, 2025 10:00 AM            ││
│ │ "Initial approval workflow"                 ││
│ │ [Rollback] [View]                           ││
│ └─────────────────────────────────────────────┘│
│                                                 │
│          [Load More (47 remaining)]             │
│                                                 │
├─────────────────────────────────────────────────┤
│ 2 selected: [Compare Versions]                 │
└─────────────────────────────────────────────────┘
```

### Diff View (Side-by-Side)
```
┌───────────────────────────────────────────────────────────────┐
│ Comparing: Version 3 ↔ Version 5                        [×]  │
├───────────────────────────────────────────────────────────────┤
│ Changes: 2 nodes added · 1 deleted · 3 modified · 2 edges +  │
├──────────────────────────────┬────────────────────────────────┤
│ Version 3 (Before)           │ Version 5 (After)              │
├──────────────────────────────┼────────────────────────────────┤
│                              │                                │
│  ┌──────────┐                │  ┌──────────┐                 │
│  │ Start    │                │  │ Start    │                 │
│  │ (gray)   │                │  │ (gray)   │                 │
│  └────┬─────┘                │  └────┬─────┘                 │
│       │                      │       │                       │
│  ┌────▼────────┐             │  ┌────▼────────┐              │
│  │ Validate    │             │  │ Validate    │              │
│  │ (yellow)    │ ← Modified  │  │ (yellow)    │ ← Modified   │
│  └────┬────────┘             │  └────┬────────┘              │
│       │                      │       │                       │
│       │                      │  ┌────▼────────┐              │
│       │                      │  │ New Check   │              │
│       │                      │  │ (green)     │ ← Added      │
│       │                      │  └────┬────────┘              │
│       │                      │       │                       │
│  ┌────▼────────┐             │  ┌────▼────────┐              │
│  │ Approve     │             │  │ Approve     │              │
│  │ (gray)      │             │  │ (gray)      │              │
│  └────┬────────┘             │  └────┬────────┘              │
│       │                      │       │                       │
│  ┌────▼────────┐             │       │                       │
│  │ Old Node    │             │       │                       │
│  │ (red)       │ ← Deleted   │       │                       │
│  └─────────────┘             │       │                       │
│                              │  ┌────▼────────┐              │
│                              │  │ End         │              │
│                              │  │ (gray)      │              │
│                              │  └─────────────┘              │
│                              │                                │
└──────────────────────────────┴────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ Parameter Changes (Validate Node)                             │
├───────────────────────────────────────────────────────────────┤
│ ▼ parameters.threshold                                        │
│   - Before: 1000                                              │
│   + After: 5000                                               │
│                                                               │
│ ▼ parameters.approval_required                                │
│   - Before: true                                              │
│   + After: false                                              │
└───────────────────────────────────────────────────────────────┘
```

### Rollback Confirmation Dialog
```
┌─────────────────────────────────────────────────┐
│ Rollback to Version 3?                          │
├─────────────────────────────────────────────────┤
│                                                 │
│ This will create a NEW version (v6) using      │
│ the workflow snapshot from Version 3.          │
│                                                 │
│ ⚠ This does NOT delete versions 4 or 5.        │
│                                                 │
│ ┌─────────────────────────────────────────────┐│
│ │ Preview:                                    ││
│ │ ┌──────┐   ┌────────┐   ┌────────┐        ││
│ │ │Start │→│Validate│→│Approve │            ││
│ │ └──────┘   └────────┘   └────────┘        ││
│ │                                             ││
│ │ 3 nodes, 2 edges                            ││
│ └─────────────────────────────────────────────┘│
│                                                 │
│ Commit message for new version:                │
│ ┌─────────────────────────────────────────────┐│
│ │ Rollback to version 3                       ││
│ └─────────────────────────────────────────────┘│
│                                                 │
│            [Cancel]  [Confirm Rollback]         │
└─────────────────────────────────────────────────┘
```

---

## Conclusion

This requirements analysis provides a comprehensive blueprint for implementing the Version History UI component. The design balances:

1. **User Experience**: Intuitive timeline, clear diff visualization, safe rollback process
2. **Performance**: Meets all Studio targets (<100ms UI, <500ms diff)
3. **Enterprise Needs**: Multi-tenancy, audit trail, permissions
4. **Technical Excellence**: TDD approach, real infrastructure testing, maintainable architecture

The progressive implementation plan (Foundation → Diff → Rollback) allows for iterative validation and reduces risk. With the backend already complete and tested (47 tests passing), frontend implementation can proceed confidently with well-defined API contracts.

**Next Steps**:
1. Review and approve requirements
2. Create ADR-0051: Version History UI Component
3. Begin Phase 1 TDD implementation (Foundation Components)
4. Conduct UX review after Phase 1 completion
5. Iterate based on feedback

**Estimated Timeline**:
- Requirements Review: 1 day
- ADR Creation: 0.5 days
- Phase 1 Implementation: 3 hours
- Phase 2 Implementation: 3 hours
- Phase 3 Implementation: 2 hours
- **Total: 8 hours development + 1.5 days planning**

This feature unlocks significant enterprise value ($30K) by providing visual version control capabilities that are critical for compliance, audit trails, and workflow management in production environments.
