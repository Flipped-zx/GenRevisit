# Agent

The agent reads structured task context and image candidates, then emits one action JSON object.

## Input

| Field | Contents |
| :--- | :--- |
| `task_context` | Original prompt, atomic constraints and image budget |
| `candidates` | Ordered candidate IDs, image artifact IDs, parent IDs, GM scores, constraint feedback and executed actions |
| `control` | Visible and editable candidates, GM-best candidate, submission candidates and remaining image budget |
| `active_skills` | Retrieved guidance and its target constraints |
| `available_skill_catalog` | Available skill IDs, versions and descriptions |
| `skill_query_control` | Completed queries, pending responses and remaining query budget |
| `last_action_error` | Optional validation feedback |

`schema_version` and `policy_id` identify the input contract. Schemas are included in `src/genrevisit/agent/schemas/`.

## Output

Each output contains `schema_version`, `action` and `arguments`.

| Action | Arguments |
| :--- | :--- |
| `query_skill` | `skill_ids`, `target_constraint_ids` |
| `generate_image` | `target_constraint_ids`, `preserve_constraint_ids`, `instruction` |
| `edit_image` | `source_attempt_id`, `target_constraint_ids`, `preserve_constraint_ids`, `instruction` |
| `submit_attempt` | `selected_attempt_id`, `reason_code` |

Generation is source-free. Editing selects a completed candidate and uses disjoint target and preserve constraints. Submission selects the observed GM-best candidate; ties use passed-constraint count and then candidate order. Scores, image bindings and control facts are supplied by the environment.

## Code

`genrevisit.agent.contract` provides input validation, action validation, control derivation, skill-query accounting and message rendering. `render_messages` binds images in candidate order and renders the system and user messages; it can append an assistant action for training.

```python
from genrevisit.agent.contract import render_messages, validate_input

validate_input(packet)
messages = render_messages(packet, image_refs)
```

Each image reference includes `attempt_id` and `artifact_id`, matching the corresponding candidate. This package covers field contracts and rendering. Model serving, image execution and the evaluation loop are separate components.
