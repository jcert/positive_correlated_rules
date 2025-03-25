# positive_correlated_rules


This repository complements the paper titled [Counterclockwise Dissipativity, Potential Games and Evolutionary Nash Equilibrium Learning](https://arxiv.org/abs/2408.00647), and accommodates Julia files for simulating a population game.

## Concept


We use system-theoretic passivity methods to study evolutionary Nash equilibria learning in large populations of agents engaged in strategic, non-cooperative interactions. The agents follow learning rules (rules for short) that capture their strategic preferences and a payoff mechanism ascribes payoffs to the available strategies. The population's aggregate strategic profile is the state of an associated evolutionary dynamical system. Evolutionary Nash equilibrium learning refers to the convergence of this state to the Nash equilibria set of the payoff mechanism. Most approaches consider memoryless payoff mechanisms, such as potential games. Recently, methods using $\delta$-passivity and equilibrium independent passivity (EIP) have introduced dynamic payoff mechanisms. However, $\delta$-passivity does not hold when agents follow rules exhibiting ``imitation" behavior, such as in replicator dynamics. Conversely, EIP applies to the replicator dynamics but not to $\delta$-passive rules. We address this gap using counterclockwise dissipativity (CCW). First, we prove that continuous memoryless payoff mechanisms are CCW if and only if they are potential games. Subsequently, under (possibly dynamic) CCW payoff mechanisms, we establish evolutionary Nash equilibrium learning for any rule within a convex cone spanned by imitation rules and continuous $\delta$-passive rules.

This repository contains the simulation used for the example in the paper, where we simulate a population selecting which path to follow in a congestion game where one of the path has a LTI delay function.   


## Requirements
- Julia 1.11


## How to use
Following the [guide on environments](https://pkgdocs.julialang.org/v1/), you can open Julia in a terminal, press `]` to access the package manager, type `activate .` and then `instantiate`. 
After installing all the required software you can press backspace to exit the package manager, now you should have all the required libraries to run the code. To run the code either use Jupyter notebook for the interactive plot or open Julia and then type `include("pc_learning_rules_new.jl")` to run the main simulation (that will generate all the simulation related figures).


## pc_learning_rules_new.jl
Runs the individual code files to generate all figures


## Aux.jl
## Dynamicsjl
## Protocols.jl
Contain functions used by pc_learning_rules.jl


