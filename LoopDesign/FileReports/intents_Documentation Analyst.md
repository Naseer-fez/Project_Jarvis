# Analysis Report for intents.py

## Dependencies
- __future__.annotations
- logging
- re
- dataclasses.dataclass
- typing.Any

## Schemas
- GoalIntentResult
- GoalIntentResult attribute: response
- GoalIntentResult attribute: mutated

## API Contracts
- parse_time_delay_with_parsedatetime(text)
- handle_goal_intent(text, user_input)
- extract_goal_delay_seconds(user_input)

## Configuration Variables
- _GOAL_CREATE_KEYWORDS
- _GOAL_STRIP_KEYWORDS
- _GOAL_LIST_KEYWORDS

## Assumptions & Notes
None

