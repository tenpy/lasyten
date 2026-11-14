# Conversion of LabelledLegs (+ label helpers)

> **Rename:** `LabelledLegs` was refactored into composed `LegLabels`
> (`Tensor.labels`). For the mechanical call-site / type replacement summary used
> by open PRs, see [convert_LegLabels.md](convert_LegLabels.md). This file keeps
> the historical batch-1–2 conversion notes.

## metadata

- original python name: `LabelledLegs` (+ helpers in same batch)
- original python file: `lasyten/tensors/_tensors.py`
- original python module: `lasyten.tensors._tensors`
- declaration in C++ header file: `include/lasyten/tensors/labels.h`
- definition in C++ file: `src/tensors/labels.cpp`
- pybind11 binding: `pybind/tensors/py_labels.cpp`
- trampoline: defer until monkey-patch (or add `PyLabelledLegs` if monkey-patching before `Tensor` is C++)
- first line of docstring: Base class that implements handling of labelled legs.

## Batch contents (order 1–2)

| Symbol | Kind | Codegen name |
| --- | --- | --- |
| `CONTRACT_SYMBOL`, `LEG_SELECT_SYMBOL`, `OPEN_LEG_SYMBOL`, `FORBIDDEN_LEG_LABEL_CHARS` | constants | manual |
| `is_valid_leg_label` | function | yes |
| `_combine_leg_labels` | function | yes |
| `_split_leg_label` | function | yes |
| `_dual_label_list` | function | **manual** (not listed by codegen) |
| `_dual_leg_label` | function | yes |
| `_get_matching_labels` | function | yes |
| `LabelledLegs` | class | yes |

## Design notes

- Label type: `using OptionalLabel = std::optional<std::string>;` — `None` ↔ `std::nullopt`.
- Label list: `using OptionalLabels = std::vector<OptionalLabel>;`
- `LabelledLegs` members: `_labels`, `num_legs`, `_labelmap` (`std::unordered_map<std::string, int64>`).
- Virtual destructor; `set_labels` virtual (overridden by `Tensor` / `ChargedTensor`).
- `get_leg_idcs`: C++ overloads for `int64`, `std::string`, `std::vector` of variants; bindings accept Python `int | str | Sequence`.
- `duplicate_entries`: small private C++ helper in `labels.cpp` (Python `tools.misc.duplicate_entries` not converted yet).
- Reuse `lasyten::to_valid_idx` from `tools.h`.
- `_get_matching_labels`: drop Python `stacklevel`; use `SPDLOG`/`logger` if available, else omit debug or call Python logging from bindings only.
- Do **not** monkey-patch `LabelledLegs` until `Tensor` is C++ (or add trampoline). Free label helpers may be monkey-patched after bindings work.

## Dependencies

- Done: `to_valid_idx`, `to_iterable` (tools; py-oriented — prefer C++ overloads for labels)
- Still Python: `duplicate_entries` (inline in C++)
- Later: `Tensor` subclasses `LabelledLegs`

## TODO list for conversion

- [x] initial setup (branch `convert_tensors`, names listed, planning docs)
- [x] planning (this file + convert_tensors.md)
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline — skipped (defer monkey-patch of `LabelledLegs`)
- [x] monkey-patch free helpers + constants; keep Python `LabelledLegs` until Tensor
- [x] run python tests with pytest (`test_tensors.py`: 4341 passed, 596 xfailed)
- [x] remove original python code for helpers/constants (class `LabelledLegs` kept)
- [x] wrap up (batch 1–2); next: `Tensor` ABC
