# Conversion note: `LabelledLegs` → `LegLabels` (composition)

Mechanical replacement summary for open tenpy / lasyten branches and PRs that still
touch the old label API. Prefer this file over
[convert_LabelledLegs.md](convert_LabelledLegs.md) (historical batch notes).

## Type / MRO

- Rename class `LabelledLegs` → `LegLabels`.
- `Tensor` / `TensorPlaceholder` **compose** a `LegLabels labels` member; they no
  longer inherit `LabelledLegs`.
- Keep `tensor.num_legs` on `Tensor` (synced with `labels.size()`).

## Label values

| Old | New |
| --- | --- |
| `OptionalLabel` / `std::optional<std::string>` | `std::string` |
| `OptionalLabels` | `std::vector<std::string>` (`LegLabelList`) |
| `std::nullopt` / Python `None` as unset | `""` |
| `if (lab)` / `lab.has_value()` | `!lab.empty()` |

## Call-site map (Python)

| Old | New |
| --- | --- |
| `LabelledLegs` | `LegLabels` |
| `t.labels` / `t.labels()` as a list | `t.labels` (`LegLabels`); copy via `t.labels.as_list()` |
| `t.set_labels(x)` | `t.labels.set(x)` or `t.labels = x` |
| `t.set_label(i, x)` | `t.labels[i] = x` |
| `t.relabel(m)` | `t.labels.update(m)` |
| `t.get_leg_idcs(x)[0]` | `t.labels.index(x)` |
| `t.get_leg_idcs([...])` | `t.labels.indices([...])` |
| `t.has_label(...)` | `t.labels.has(...)` or `'x' in t.labels` |
| `t.labels_are(...)` | `t.labels.are(...)` |
| `t.is_fully_labelled` | `t.labels.is_fully_labelled` |
| `t._labels` / `t._labelmap` | remove (compare `t.labels` or use `as_list()`) |
| `for x in t.labels` / `list(t.labels)` | `for x in t.labels.as_list()` / `t.labels.as_list()` |
| `assert t.labels == [a, None, b]` | `assert t.labels == [a, '', b]` |

`LegLabels` supports equality, `len`, `[]` get/set (int or str key), and
assignment from a sequence. It does **not** implement `__iter__`.

`relabel1` / `relabel2` kwargs on `compose` / `tdot` / `outer` / etc. are unchanged
(function arguments, not tensor methods).

Chaining note: old `t.relabel(m)` returned `self`; `t.labels.update(m)` returns
the `LegLabels` member. Split chains:

```python
# old: expect = move_leg(...).relabel({...})
expect = move_leg(...)
expect.labels.update({...})
```

## HDF5

- Nested `"labels"` stores a string list; load old `None` entries as `""`.
- Standalone roundtrips: `LegLabels::{save_hdf5,from_hdf5}`.

## Bindings / stubs

- `pybind/tensors/py_labels.cpp`: bind `LegLabels`; sequences of `str` only.
- Drop nested `labels` property and `_labelmap` from the old holder API.
- `lasyten/_core.pyi`: `list[str]`, not `list[str | None]`.
