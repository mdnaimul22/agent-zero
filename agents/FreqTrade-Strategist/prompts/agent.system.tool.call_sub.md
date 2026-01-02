### available subordinate default profiles
Arg usage:
profile: select from available profiles for specialized subordinates, leave empty for default
message: "always describe task details goal overview for new subordinate",

reset:
  "true": spawn new subordinate
  "false": continue existing subordinate
if superior, orchestrate

respond to existing subordinates using call_subordinate tool with reset false

example usage
~~~json
{
    "thoughts": [
        "The result seems to be ok but...",
        "I will ask a coder subordinate to fix...",
    ],
    "tool_name": "call_subordinate",
    "tool_args": {
        "profile": "Backend-Developer",
        "message": "...",
        "reset": "true"
    }
}
~~~

**response handling**
- you might be part of long chain of subordinates, avoid slow and expensive rewriting subordinate responses, instead use `§§include(<path>)` alias to include the response as is

**available profiles:**
{{agent_profiles}}