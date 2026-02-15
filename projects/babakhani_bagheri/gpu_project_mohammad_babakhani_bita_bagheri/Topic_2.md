# Topic 2 — N-Body Simulation Using CUDA (Solar System with a Comet)

## What is an N-Body problem?

An **N-body problem** refers to the simulation of a system of **N interacting bodies**, where each body interacts with every other body according to some interaction rule or force model. While N-body problems often appear in **gravitational systems**, the concept is more general and applies to any system with pairwise interactions (e.g., physical forces, particle systems, or abstract interaction models).

The key challenge of the N-body problem is that:

* Every body interacts with every other body
* The computational complexity grows as **O(N²)**

This makes the N-body problem a **classic performance benchmark** and a natural fit for GPU programming.

## Goal

In this project, **you should implement a general N-body simulation using CUDA**, with a **primary focus on performance and scalability**.

The goal is **not limited to gravitational physics**. Instead, the emphasis is on efficiently computing and updating large numbers of **pairwise interactions** on the GPU. The interaction model you choose (gravitational or otherwise) should be clearly defined and justified in your proposal.

You should treat this project as a **parallel numerical simulation problem**, where performance considerations drive your design decisions.

## Project idea: Solar system with a comet

This scenario is provided **as an example to give you insight into the nature of N-body problems**. You are encouraged to **define your own N-body system and problem setting** in your proposal.

One example system is a simplified **solar system** consisting of:

* The Sun
* Multiple planets and planetoids
* One or more comets

Each body should:

* Have a state (e.g., position, velocity, or other attributes)
* Interact with other bodies according to a defined interaction rule
* Be influenced by the gravitational force of all other bodies

In this system, you should carefully consider **relative mass scales**. In particular:

* The Sun’s gravitational influence should be dominant
* Small planetoids should **not significantly disturb planetary orbits**
* Planetary motion should remain stable over long simulations

The simulation should demonstrate meaningful physical behaviors such as:

* Stable orbital motion of planets around the Sun
* Limited influence of small bodies on large ones
* A comet trajectory that is **distorted, deflected, or slingshot** by a planet’s gravitational field

The emphasis is not on astronomical precision, but on correctly modeling **pairwise interactions**, numerical stability, and efficient time evolution.

## Performance-centered focus

You should focus on both:

* **Correctness** — bodies move in a physically reasonable way and conserve basic properties over short timescales
* **Performance (primary focus)** — how efficiently pairwise interactions are computed

Your design should explicitly consider:

* Mapping pairwise force computation to GPU threads
* Memory access patterns when reading body data
* Minimizing global memory traffic
* Reducing redundant computations
* Scaling behavior as the number of bodies increases

You should clearly explain why your CUDA design performs better than a naïve implementation.

## What “from scratch” means

You should implement:

* CUDA kernels to compute pairwise interactions
* Position and velocity updates over discrete time steps
* The full simulation loop on the GPU

You may use:

* **Simple preprocessing on the CPU** (initial conditions, normalization, input generation)
* CUDA runtime and standard CUDA libraries
* CPU-side code for visualization or data output

You should **not** use existing physics engines or GPU-accelerated N-body libraries.

## Proposal requirement

Before starting implementation, **you should submit a short proposal (1–2 pages)** describing:

* The scale of your simulation (number of bodies)
* How you model interactions between bodies
* Your CUDA parallelization strategy
* Expected performance bottlenecks
* How you plan to evaluate scalability and performance

You should **present this proposal** and justify your performance-related decisions.
If the proposal is **approved**, you may proceed with the full implementation.

## Implementation expectations

Once approved, you should:

* Implement CUDA kernels for interaction computation and state updates
* Run the simulation for many time steps
* Demonstrate stable large-scale behavior and meaningful interactions (e.g., comet deflection)
* Measure performance as the number of bodies increases

Simulations of this type typically benefit from a **real-time visual interface**. You are encouraged to include a simple UI or visualization layer that allows you to:

* Observe the system evolving over time
* Verify physical behavior visually
* Interact with the simulation parameters if desired

The simulation should aim to run **in real time or near real time**, reinforcing the importance of GPU performance.

## Evaluation and reporting

Your final report should include:

* Visual or numerical evidence of correct system behavior
* Performance measurements (time per step, interactions per second)
* Scaling analysis as N increases
* A discussion of the main performance bottlenecks and how you addressed them

Your work should clearly demonstrate how GPUs excel at large-scale pairwise interaction problems, and how CUDA programming choices impact simulation performance.
