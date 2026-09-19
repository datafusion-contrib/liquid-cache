## Write code that requires minimal human attention 

- Fact: every line of code requires human review, so write readable code not just working code.

- Build reusable, trustworthy components that has deep logic but thin interface, so that human only need to review the complexity once (reuse the attention).

- Follow Rust best practices to save human attention from reviewing for common mistakes.

- Before implementing a new feature, identify similar existing features, and check if it's worth to refactor existing code so it's reusable.

## Engineering

- Prefer explicit props and function arguments so ownership and data flow remain visible. Use context only when a dependency must cross at least two component layers and the intermediate parent does not use it. Keep state owned by the component that creates it.


- use descriptive, intention-revealing names for variables and functions, don't use generic names like `Built`, `Common`, `View`.

- do not use free form functions, functions should have a clear owner, e.g., struct, enum; unless it is fully justified stateless function.
  Example 1: `fn build_chart(model: &DataModel) -> Built`, instead use a member method like: `fn build_chart(&self) -> Built`. Even better, use a `From` trait to build the chart from the data model.
  Example 2: `fn toggle_expand(code: crate::views::codemap::CodeState, key: (u32, u32))` instead use a member method like: `fn toggle_expand(&self, key: (u32, u32))`.

- try to use minimal visibility for functions and variables, e.g., a pub function under a private mod is essentially private, but it is confusing, prefer to make it private. 

- try to avoid public fields, prefer to use getters and setters, unless the field is truly public.

- organize the code by features, not by technical buckets, e.g., don't have `api.rs`, `views.rs`, `data.rs`, etc. Instead, do `settings.rs`, `data_panel.rs`, `code_map.rs`, etc.

- No static variables.

- Don't use large unconstrained struct constructions, e.g., following code is ugly and error prone.
Maybe there're invariants we want to enforce, but this code allows any fields to be set.
```rust
ghost_nodes.push(CrateInfo {
                            id: ghost_id.clone(),
                            name: ev.name.clone(),
                            version: ev.detail.clone().unwrap_or_default(),
                            is_member: false,
                            changed: false,
                            changed_files: 0,
                            manifest_changed: false,
                            affected_dist: None,
                            dependents: 0,
                            direct_deps: 0,
                            external_deps: 0,
                            ghost: true,
                            description: None,
                            license: None,
                            repository: None,
                            homepage: None,
                            documentation: None,
                            // A removed dependency's manifest is gone with
                            // it; the name is all we know.
                            crates_io: false,
                            rel_path: None,
                        });
```
Instead, we use use a much narrower constructor, with new(arg1, arg2, arg3), and check and enforce the invariants.

- Typically a function should not have more than 3 parameters (including self if it is a member method). If it does, it is a sign of either too large function body, or a container struct should hold the parameters. 

- A struct should not have more than 7 fields, more than that adds cognitive burden. Use private structs to group related fields.
