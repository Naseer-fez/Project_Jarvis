# Analysis Report for permission_matrix.py

## Dependencies
- __future__.annotations
- dataclasses.dataclass
- dataclasses.field

## Schemas
- PermissionResult
- PermissionResult attribute: blocked_actions
- PermissionResult attribute: confirmation_actions
- PermissionMatrix

## API Contracts
- PermissionResult.has_blocked(self)
- PermissionResult.needs_confirmation(self)
- PermissionMatrix.__init__(self, config)
- PermissionMatrix.evaluate(self, actions)
- PermissionMatrix._parse_csv(self, section, key)

## Configuration Variables
None

## Assumptions & Notes
- Module Docstring: Compatibility permission matrix built on top of the risk config.

