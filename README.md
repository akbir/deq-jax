# Deep Equilibrium Models [NeurIPS'19]
Jax Implementation for the deep equilibrium (DEQ) model, an implicit-depth architecture proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun.

Unlike many existing "deep" techniques, the DEQ model is a implicit-depth architecture that directly solves for and backpropagates through the equilibrium state of an (effectively) infinitely deep network. 

## Prerequisite
Python >= 3.5 and Haiku >= 0.0.2

## Major Components
This repo provides the following re-usable components:

1. Jax implementation of the Broyden's method, a quasi-Newton method for finding roots in k variables. This method is JIT-able.
2. Jax implementation of DeepEquilbrium's Rootfind method with custom vector-Jacobi product (backwards method)

## Credits
The repo takes direct inspiration from the [original implementation](https://github.com/locuslab/deq/tree/master) by Shaojie in Torch. The transformer module is taken from the Haiku example.