# JAX JIT Compilation

Date read: 23rd September 2025

[Blog link](https://docs.jax.dev/en/latest/jit-compilation.html)

## Key Points
* JAX reduces function into a set of primitive functions (fundamental unit of computation).
* These can then be compiled.
* Important points:
  * JAX functions must be **functionally pure**... i.e. have no side effects (e.g. edit global variables) as these will not be recorded
  and won't throw error messages. Print functions are included in this.
  * Code is run once to trace it (hence name Just After eXecution), but this should not be used for any functionality as it is an implementation detail.
  * The above means calling a function for the first time is slow.
  * **Control flow cannot be controlled by dynamic runtime values**: (if statements and while loops)
    * Write without conditional
    * Use control JAX flow operators like jax.lax.cond()
    * JIT compile the non-conditional parts and then call them when their condition is met.
    * Set static_argnums to not include conditional value and it will use a less abstract tracer
    * (Static = value known at compile time, dynamic = not known at compile time)
* Tips:
  * Wrap as much code as possible in JIT compile to give the compiler more code to work with... it will optimise it further and reduce the call overhead.
  * **JAX caches previous compilations and reuses it for future calls**, if we use static argnums it will reuse it for the same values labelled as static.
  This is helpful when theres a small set of values an input could take.
