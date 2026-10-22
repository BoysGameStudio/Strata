# Recovering an unknown tool name

A model may produce a complete `bash` tool call when the client only declared
`shell`. By default Strata returns that block as text. An agent may then stop,
without receiving a tool error that it could use to correct its next call.

This fork adds an opt-in server setting for clients that validate tool names
and return unavailable-tool errors to the model:

```json
{
  "undeclared_tool_policy": "passthrough"
}
```

Add this key to the existing server JSON config, alongside `tool_call_recovery`,
not inside the engine `args`. Restart the server after editing it. The default
is `"text"`; removing the key or setting it to `"text"` restores the old response
behavior. Other values are rejected before the inference engine is created.
This setting is not a per-request API option or an upstream release feature.

## What changes

With `passthrough`, an eligible unknown call is delivered with its original name
and a call ID. The client must reject it, return a tool error with that ID, and
request another turn. Strata does not rename `bash` to `shell`, register a tool,
execute a command, retry generation, or bypass the client's permission checks.

Only complete standard `<tool_call><function=NAME>...` envelopes in the answer
are eligible. Unknown calls are buffered until both closing tags arrive; no
partial tool events are sent. Malformed or cut calls stay text. The additional
buffer is limited to 1,048,576 characters plus closing-tag lookahead, not tokens.
A name must contain 1–128 ASCII letters, digits, underscores, dots or hyphens.
Arguments have no declared schema for an unknown name, so their types are parsed
by the existing schema-free parser. The client must reject the name before any
execution; this feature does not promise that its arguments match another tool.

The service only enables the policy when tools are declared and `tool_choice`
is absent or `auto`. Forced, required, none and unsupported choices retain the
existing behavior. Responses API compatibility, server-side MCP requests,
reasoning, quoted code, JSON recovery, bare functions and drifted unknown openers
are not broadened by this setting. Declared tools keep their existing parser and
streaming path. The standard `_callable()` check is not relaxed.

The policy applies to eligible requests from **all clients of that server**, not
just OpenCode. Leave the default in place when a client does not safely reject
unavailable tools. Use a separate server/profile when clients need different
policies. Existing tool-choice and recovery behavior outside this feature is not
redesigned here.

## Diagnostics

For a standard unknown opener the parser records `undeclared_as_text` or
`undeclared_passthrough` and logs the outcome. Logs include only a bounded tool
name, not arguments or conversation text. The older `unread` counter keeps its
existing meaning; it is not an unknown-tool counter.

The startup message confirms the enabled policy. A message saying an unknown
call was passed to the client proves only delivery, not execution or recovery.

## Verification

Run the CPU-only tests from the checkout using its Python environment:

```sh
python -m unittest serve.test_tool_call_envelope serve.test_undeclared_tool_policy -v
python -m unittest serve.test_server serve.test_structured -v
```

The new tests cover the parser's streaming/recovery matrix, incomplete input,
quoted examples, malformed names and envelopes, buffer boundaries, request
policy gates, and real localhost HTTP/SSE responses over a scripted engine.
A two-request protocol test sends a client tool error and then receives a valid
`shell` call. It does not run shell commands and is not a real-model or OpenCode
integration test.

For deployment acceptance, verify with the actual OpenCode instance that the
unknown call is rejected, its error is fed back, and a later valid call executes
at most once. Real-model correction rate, GPU performance and that OpenCode
acceptance check are separate from these CPU tests.
