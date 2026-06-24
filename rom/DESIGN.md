# CEQP: design choices

This document records the main design choices behind the
conservative EQP (CEQP) hyperreduction implemented in this directory.
It is meant as a durable, human-readable picture of what we chose
and why, without references to the underlying code.

## Overview

CEQP is an energy-conserving variant of Empirical Quadrature
Procedure (EQP) hyperreduction for the reduced-order model of
Lagrangian compressible hydrodynamics. Basic EQP (BEQP)
selects a sparse set of quadrature points and weights so that the
reduced force is reproduced accurately, but it does not, by itself,
preserve the discrete total energy of the reduced system. Over long
time horizons and across time-window basis changes this shows up as a
slow energy drift.

The goal of CEQP is to match BEQP's accuracy while conserving the
total energy (kinetic plus internal) of the reduced model to machine
precision. Every design choice below exists to serve that goal without
sacrificing accuracy or adding meaningful online cost.

### Simulation stages

For orientation, the model is built and run in five stages, and the
choices below are distributed across them.

- **Offline**: full-order simulations generate state snapshots.
- **Merge**: snapshots are compressed into windowed reduced bases.
- **Prep**: the bases are enriched and orthonormalized, and the
  hyperreduction (the sparse quadrature rule) is built.
- **Online**: the hyperreduced model is time-stepped.
- **Restore**: the reduced trajectory is lifted back to full order
  and compared against the full solution.

Each of the conceptual choices that follow touches several of these
stages, which is why we organize the rest of the document by
principle rather than by stage.

## The conservation principle

Discrete energy conservation in the reduced model is not automatic; it
holds only when a few structural conditions are met simultaneously.
CEQP is, in essence, the set of choices that enforce all of them at
once:

1. The velocity and energy force terms must be evaluated with the
   *same* quadrature rule, so that the work done by the force on the
   velocity equation exactly cancels the corresponding term in the
   energy equation.
2. The velocity offset used to define the reduced coordinates must be
   zero, so that no spurious energy source is injected.
3. The time integrator must have an averaging structure that respects
   the energy balance at the discrete level.
4. The reduced bases must be compatible with the mass-weighted inner
   product in which energy is naturally measured.

The remaining sections describe the concrete choices that satisfy
these requirements, and then how energy is preserved across time
windows.

## Mass-orthonormal bases

We orthonormalize the velocity and energy bases with respect to their
mass-matrix inner products, so that the reduced mass matrices are the
identity. Energy is a mass-weighted quantity — kinetic energy is the
mass-weighted norm of the velocity, and internal energy is a
mass-weighted integral of the specific internal energy — so measuring
and conserving it is cleanest in a basis that is orthonormal in that
same inner product.

This choice pays off in three ways. First, conservation becomes exact
rather than approximate: there is no reduced mass matrix to invert
online, and therefore no round-off error from a numerical inverse
accumulating step after step. Second, the kinetic energy and internal
energy of the reduced state are read directly from the reduced
coordinates, which makes both the energy diagnostic and the
cross-window correction (below) simple and basis-independent. Third,
the online integrator needs no special handling: because the reduced
mass matrix is the identity and the raw basis appears in the force
evaluation, the hyperreduced force evaluation already produces the
reduced acceleration directly.

The mass-orthonormalization is performed once per window during the
prep stage, using a mass-induced Gram-Schmidt process. The
corresponding mass-weighted projection of the full state onto the
reduced basis is likewise an offline operation; it never appears on
the per-step online path.

## Energy-unit enrichment

We append a constant "energy-unit" field---the all-ones vector in the
energy space---to the energy basis before orthonormalization. For the
piecewise-constant (partition-of-unity) energy discretization used
here, this vector represents a uniform unit of specific internal
energy across the domain.

The reason is that the total internal energy of the reduced state is
the mass-weighted inner product of the energy field with this unit
field. By forcing the unit field to lie exactly in the span of the
energy basis, the reduced model represents the internal energy exactly,
and---as explained later---preserves it exactly across window changes as
well. Without this enrichment the internal energy would only be
captured up to the truncation error of the data-driven basis.

This enriched, then orthonormalized, energy basis is the one used
throughout the online run.

## Combined velocity-energy quadrature rule

In BEQP one builds separate sparse quadrature rules for
the velocity force and the energy force, each tuned to reproduce its
own quantity as accurately as possible. CEQP instead solves a *single*
non-negative least-squares problem for a combined velocity-energy force
target and assigns the resulting rule to both the velocity and the
energy force evaluations.

This is the discrete embodiment of the first conservation condition:
because the two force terms are integrated with identical weights at
identical points, the energy injected into the velocity equation and
the energy extracted in the internal-energy equation cancel exactly.
A rule that was merely accurate for each term separately would leave a
small, systematically signed residual that drives the long-time energy
drift. The combined rule is what removes it.

## Time integration

The reduced model must be advanced with an averaging Runge-Kutta
integrator (the RK2-average scheme), not the default fourth-order
Runge-Kutta. The discrete energy balance that the rest of the design is
built to satisfy relies on the specific averaging structure of this
integrator; it does not hold for a general explicit scheme. Both BEQP
and CEQP runs therefore use this integrator so that the comparison
between them is meaningful and so that CEQP's conservation property is
realized in practice.

## Offset handling

Reduced-order models for this class of problems are usually built on
*deviations* from a reference state (an offset) rather than on the raw
fields, because the reduced basis then only has to resolve how the
solution moves away from a known starting point. Keeping an offset
active during the offline and basis-construction stages is important
for accuracy: it lets the data-driven basis spend its degrees of
freedom on the dynamics rather than on representing a nearly constant
background state.

However, a nonzero *velocity* offset violates the second conservation
condition: it introduces a spurious source into the reduced energy
balance. Position and energy offsets are harmless in this respect, but
the velocity offset is not.

CEQP resolves this tension by *absorbing* the offset into the bases.
Each window's offset---its own starting state---is appended as an extra
basis column and carried through the same mass-orthonormalization as
the rest of the basis. The online run is then performed with no offset
at all: the absorbed columns let the offset-free reduced model still
represent the initial fields of each window, while the velocity offset
is structurally zero. This gives accurate, deviation-based bases
offline and an energy-clean, offset-free model online, for any
problem---including those with a nonzero initial velocity, where
a subtract-the-offset approach would break conservation.

Note that absorbing the energy offset is a separate act from the
energy-unit enrichment described earlier: the unit field is a uniform
constant, whereas the absorbed offset is the actual initial
internal-energy distribution. Both are needed.

## Multi-window continuity and energy preservation

Long simulations use a sequence of time windows, each with its own
reduced basis. At a window boundary the reduced state must be handed
from the old basis to the new one. We do this by *re-coordinatizing*:
the new reduced coordinates are obtained by projecting the old state
onto the new basis through a mass-weighted transition operator. Because
energy lives in the mass-weighted inner product, this transition must
itself be mass-weighted; a plain Euclidean projection would be
inconsistent with mass-orthonormal bases. The transition operator is
small and is precomputed offline, so the online cost at a window change
is a single reduced matrix-vector product.

Re-coordinatization alone, however, is not energy-conserving. The new
basis cannot represent the component of the outgoing velocity that lies
outside its span, so that component is dropped and the kinetic energy
takes a small downward jump at every window boundary. The internal
energy does not jump, precisely because the energy-unit field lies in
every window's span. The entire energy loss at a transition is
therefore kinetic.

We remove this jump with a *kinetic-energy renormalization*. After
projecting the velocity onto the new basis, we rescale the projected
velocity coordinates by a single scalar so that their norm returns to
the value it had before the transition. Since kinetic energy is exactly
the squared norm of the velocity coordinates in our mass-orthonormal
basis, this restores the kinetic energy---and hence the total
energy---to its pre-transition value, to round-off. Internal energy
is left untouched. The correction costs essentially nothing and,
in practice, changes the solution negligibly while removing the
energy jump entirely.

### A noted alternative

Kinetic-energy renormalization conserves energy but does not recover
the actual velocity component that the new basis fails to represent; it
simply rescales what the basis can represent. If, in the future, the
accuracy of the *state* at window boundaries (rather than just the
energy) turns out to matter, a more elaborate option is available:
enrich the receiving window's basis with the outgoing solution itself,
so that the dropped direction is represented exactly. This would
conserve energy too, and additionally capture the boundary state, at
the cost of materially more machinery.

It is worth noting that this enrichment generalizes naturally to the
predictive setting. The enrichment direction is the live outgoing
solution at the transition, reconstructed from basis-only quantities
that can be precomputed offline together with the reduced coordinates
that the online run already carries; it does not require knowing the
boundary state in advance. The price is that the enrichment column, and
the consequent growth of the reduced dimension by one per transition,
must be assembled and propagated in the online stage rather than baked
in offline. We have deliberately chosen the lightweight renormalization
for now, since our present goal is energy conservation rather than
boundary-state accuracy, and we record the enrichment approach here as
the natural next step should that goal change.

## Scope

Our testing to date targets *reproductive* simulations, but the
conservation machinery itself is not tied to that setting. Every
precomputed object above---the mass-orthonormalization, the combined
quadrature rule, the absorbed offsets, and the cross-window transition
operators---depends only on the reduced bases and the mass matrix, not
on the full solution at the window boundaries. The cross-window
correction likewise reads only the live reduced coordinates that the
online run already carries. These objects are therefore precomputable
offline and applicable to any online trajectory, in or out of sample.

A useful consequence is that energy conservation is *structural*: it
follows from the matched quadrature, the zero velocity offset, the
averaging integrator, the mass-orthonormal bases, and the cross-window
renormalization, none of which assume the online trajectory matches the
training data. CEQP therefore conserves total energy to machine
precision even on out-of-sample runs.

What remains reproductive is *accuracy*, not conservation: the bases
and the quadrature rule are built from training snapshots, so the
approximation degrades as a trajectory leaves the training manifold.
This is the generic limitation of any projection-based reduced model
and is independent of the energy-conservation property---an
out-of-sample run may be inaccurate yet still conserve energy exactly.

## Command line flags

The choices above map onto the following command line flags, used
across the offline, merge, prep, online, and restore stages. This is a
pointer for orientation, not a full usage reference.

- `-s 7` — select the RK2-average integrator (required for
  conservation; the default `-s 4` is fourth-order Runge-Kutta and does
  not conserve).
- `-romos` with `-rostype load` — keep the per-window offset active
  offline so the bases resolve deviations from each window's own
  starting state.
- `-no-romoffset` — run the online and restore stages offset-free,
  relying on the absorbed offset columns.
- `-no-romgs` — mass-orthonormalize the bases offline rather than
  applying Gram-Schmidt online.
- `-romsns` — skip storing the separate velocity/energy force
  snapshots, since the combined quadrature rule does not need them
  (must be set consistently across stages).
- `-maxnnls` — iteration cap for the non-negative least-squares solve
  that builds the quadrature rule; raise it if the combined rule does
  not converge.
- `-lqnnls` — enable LQ preconditioning of that solve.

The hyperreduction is built in the prep stage (`-online -romhrprep`)
and exercised in the online stage (`-online -romhr`).
