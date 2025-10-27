# GCC 13 build errors and fixes

This repository failed to build with GCC 13 (C++20) due to two independent issues:

1) constexpr misuse of `std::string`
2) Overly broad operator templates hijacking `std::vector` internals

Below is a concise explanation and the changes applied to fix both.

## 1) `constexpr std::string` is ill-formed in C++20

Error excerpt:

```
/usr/include/c++/13/bits/allocator.h:195:52: error: ‘std::string("EstimationTechnique")’ is not a constant expression because it refers to a result of ‘operator new’
```

Root cause:
- `std::string` is not a literal type in C++20; creating one in a `constexpr` context requires dynamic allocation (disallowed in constant evaluation).
- We had multiple declarations like:
  - `static constexpr std::string ... = "literal";`
  - `constexpr std::string algo_type() { return "..."; }`

Fixes:
- Replace `constexpr std::string` constants with `constexpr std::string_view` (or `const char[]`), which are valid at compile time.
- Change `algo_type()` to return `constexpr std::string_view`.
- Added `#include <string_view>` where needed.

Files affected:
- `src/utility/traits_fdagwr.hpp` (many constants and `algo_type()`)
- `src/fwr_predictor/fwr_predictor.hpp` (predictor ID constants)
- `src/basis/basis_include.hpp` (basis name constants and array)

Why this is safe:
- `std::string_view` refers to string literals without allocation. If a `std::string` is required at a call site, construct it from the `string_view` (runtime, not constexpr).

## 2) Global operator overload captured `end() - 1` inside `std::vector`

Error excerpt:

```
/usr/include/c++/13/bits/stl_vector.h:1236:16: error: no match for ‘operator*’ (operand type is ‘SubExpr<...>’)
 1236 |         return *(end() - 1);
```

What happened:
- We had generic operator templates (e.g., `operator-`, `operator*`, `operator+`) for functional matrix expression templates in the global namespace.
- Their constraints were too broad (basically "not Eigen"), so `end() - 1` inside libstdc++ chose our global `operator-` instead of iterator subtraction, producing a `SubExpr` instead of an iterator.
- The subsequent unary `*` then failed, because the result wasn’t an iterator anymore.

Fix:
- Introduced a tighter concept `fm_utils::functional_matrix_like` that requires the presence of `rows()`, `cols()`, `size()`, and `operator()(i,j)` — characteristics of our functional matrix expressions.
- Updated all relevant operator templates (`+`, `*`, binary `-`, unary `-`, plus `exp` and `log`) to require `functional_matrix_like<...>` instead of just `not_eigen<...>`.

Files affected:
- `src/functional_matrix/functional_matrix_utils.hpp` (new `functional_matrix_like` concept)
- `src/functional_matrix/functional_matrix_operators.hpp` (updated `requires` clauses)

Why this is safe:
- Constraints now precisely select our expression types and won’t interfere with STL iterator arithmetic or unrelated types, while preserving the intended overloads.

## Summary of changes

- Use `std::string_view` for all compile-time string constants; include `<string_view>`.
- Return `std::string_view` from `algo_type()`.
- Add concept `fm_utils::functional_matrix_like` and use it to constrain operator overloads.
- No public API behavior change for typical usage; compile-time safety and GCC 13 compatibility are improved.

## Follow-ups (optional)
- Consider moving operator overloads into a dedicated namespace that also contains the expression types, relying on ADL. This further reduces the chance of global overload clashes.
- If any call sites depend on owning `std::string`, construct `std::string{constant_sv}` as needed.
