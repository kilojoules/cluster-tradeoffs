# Blob Discovery Goals

## Core Question

what is the maximum regret we can create for a wind farm cluster by controlling the neighboring wind farm characteristics?

In this set up, we have a wind farm cluster. This is defined in terms of a target wind farm, whose positions are to be optimized, and neighboring wind farm(s). 

Regret is defined as follows:

1. optimize target wind farm with multistart (e.g., 10 random initial guesses), assuming no neihgbors and just modeling the turbines in isolation
2. optimize target wind farm with multistart (e.g., 10 random initial guesses), assuming the neihgbors are present and taking their weights into accoint
3. Pool all 20 layouts. Evaluate AEP with and without neighbors. These can be used to construct a pareto set.

If there is only one pareto point, there is zero regret. If there is more than one pareto point, regret is defined as conservative AEP of the conservative AEP optmimal layout minus conservative AEP of liberal AEP optimal layout. 

Since we are approximating global maximimum, we should ensure we are using a sufficient amount of SGD iterations

## The Problem

We need to agree on design variables to describe the cluster characteristics. 

There is a two-layer optimization here. In the inner optimization, we do the regret computaiton described above. Ideally we can get gradients of this regret w.r.t. neighboring wind farm and/or cluster design variables. This may or may not include the wind rose characteristics. 

# The headline

through this work we identify "worst case" cluster configurations that will lead to the most regret when not designing with neighbors in mind
