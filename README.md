<div align="center">
    <h1><code>expreon</code></h1>
    <i>Eevee + AST? = Expreon!!</i>
    <br/>
     A set of crates to implement Symbolic Regression using Genetic Programming (GP) in rust.
    <hr>
</div>

> NOTE: This is a work in progress, I'm implementing this as the GP engine for another project.

This library implements a set of primitives to implement Symbolic Regression using Genetic Programming (GP) techniques in rust in the form of multiple independent reusables crates:

- `crates/expreon_ast`: Provides the arena-allocated AST expression machinery (independent of the evaluation engine)

- `crates/expreon_eval`: Provides the evaluation runtime for expressions, operations and a set of operations built-in.

- `crates/expreon`: The library crate that wraps both crates and provide GP primitives to implement SR along with a library of mutations built-in.


## Where does that name come from?????

I like eevee (the cute pokemon), and eevee is a pokemon that evolves a lot (DNA mutations) -> like in GP where individual solutions are mutated and bred. Hence the name expreon.


## Acknowledgments

This is inspired by the [Operon](https://github.com/heal-research/operon) C++ framework which implements Symbolic Regression with optimization of model coefficients using the Levenberg-Marquardt algorithm.

