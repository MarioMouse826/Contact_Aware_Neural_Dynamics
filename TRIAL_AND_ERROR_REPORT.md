# Trial And Error Report

## Objective

The goal of this project iteration was to train a simple robot hand that could reliably:

1. approach the object at location A
2. grasp the object
3. lift it
4. move it to location B
5. place it at the correct position
6. release it cleanly

The hardest part was not the initial grasp or lift. The main difficulty was getting the full sequence to happen smoothly and consistently, especially:

- avoiding dragging instead of lifting
- avoiding shaking near location B
- getting the hand to actually release
- reducing turbulent up-and-down motion during transport

## Starting Point

The project began from the contact-aware SAC setup in this repository. Early versions could often:

- reach the object
- establish contact
- sometimes grasp and lift

But they struggled to complete the full pick-place-release sequence.

Two important historical reference runs emerged during this process:

- `3rfnpbbk`
  - strong, sharper motion
  - smoother approach and transport behavior
  - good lift and move dynamics
  - weak or missing final release behavior

- `3hq3kpce`
  - learned the full objective end-to-end
  - could grasp, lift, move, place, and release
  - still showed roughness and transport turbulence

These two runs shaped most later decisions.

## Main Problems Observed

Across training runs and videos, we repeatedly observed the following failure modes:

### 1. No grasp or unstable grasp

Some runs showed:

- `grasp_rate = 0`
- weak contact stability
- long episodes with no meaningful interaction

This suggested the policy either was not finding the object well enough or was exploiting unstable contact without forming a useful grasp.

### 2. Sliding instead of lifting

In several versions, the hand moved the object sideways toward B before lifting it. This meant the reward still gave value for transport-like progress even when the object had not been properly lifted.

### 3. Hovering or shaking above B

Some policies learned:

- grasp
- lift
- move toward B

but then hovered, shook, or oscillated above the goal instead of descending and finishing the place.

### 4. Refusing to release

Even after successful placement, some runs:

- held onto the object
- partially opened but re-contacted
- timed out while still gripping

This showed that release was under-incentivized or too late in the reward sequence.

### 5. Rough vertical motion during transport

Later successful models could finish the task, but the carry phase still had unwanted up/down motion. This became the main refinement problem once end-to-end success was achieved.

## Major Changes Made

The following sections summarize the meaningful reward, control, and training changes we made.

## Phase 1: Build A Stronger Pick-Place Reward Sequence

### Change: preserve lift value during transport

We changed the reward so that once the policy achieved a valid lifted transport grasp, it would not lose carry progress while moving toward B or beginning descent.

Reason:

- earlier shaping could make the agent prefer hovering or preserving lift over completing the task
- transport needed to remain valuable after lift was achieved

Effect:

- improved transition from lift to move
- reduced cases where moving toward B was accidentally discouraged

### Change: transport reward requires lift

We added logic so transport reward would not activate unless the object had actually been lifted enough for transport.

Reason:

- the model was learning to drag or slide the object toward B

Effect:

- stronger task ordering
- better separation between grasp/lift and move phases

### Change: stronger place shaping near B

We improved the placement reward so the policy received more meaningful signal while descending over B instead of only near the final strict placement condition.

Reason:

- the model often hovered above B without finishing the downward motion

Effect:

- better descent behavior
- less incentive to remain suspended above the goal

## Phase 2: Make Release Explicit

### Change: release corridor reward

We introduced a release corridor concept for the pick-place task. This rewarded the following sequence near B:

1. reach the goal zone
2. lower into a valid placed position
3. open fingers
4. move away

Reason:

- simply rewarding fully released states was too sparse
- the hand often kept gripping because release only paid off at the very end

Effect:

- the model started learning actual release behavior
- `3lg85vtz` was one of the first versions that visibly learned to release

### Change: pre-release opening bonus

We added reward for beginning to open near the goal before full detachment.

Reason:

- release needed a progressive signal, not just a binary success condition

Effect:

- improved willingness to open fingers once over B

### Change: post-release retreat bonus and re-contact penalty

We added:

- a reward for retreating after release
- a penalty for returning to contact after release

Reason:

- some policies opened the hand but then moved back toward the object
- others hovered too close and re-contacted unnecessarily

Effect:

- cleaner end-stage behavior
- more decisive completion of the release phase

## Phase 3: Improve Stability And Remove Unnatural Behavior

### Change: stricter grasp interpretation

We improved grasp-related logic to avoid rewarding awkward or unstable contact patterns too easily.

Reason:

- some runs exploited contact in unrealistic ways
- a cube could appear unnaturally stuck in the hand

Effect:

- better alignment between reward and useful grasps

### Change: action smoothness penalties

We used:

- `action_penalty_weight`
- `action_delta_penalty_weight`
- `lift_instability_penalty_weight`

to discourage jittery control.

Reason:

- several policies produced shaky lifting or unstable corrections

Effect:

- reduced extreme twitching
- improved smoothness in some stages

Important lesson:

- too much smoothing made the hand too passive and caused failures in lift or transport
- therefore smoothing had to be used carefully

## Phase 4: Separate Good Ideas From Bad Ones

Several attempted fixes looked promising at first but were rolled back because they created worse behavior.

### Attempt: Better_Transitting&Forcing_Release

This version tried to:

- force release more aggressively
- reduce clinging at the goal
- smooth behavior

Why it was rolled back:

- the hand became too slow
- some runs lost useful lift and transport behavior
- the policy sometimes failed earlier in the sequence

Lesson:

- stronger release shaping can help, but if it dominates too early it can damage the rest of the task

### Attempt: overly strong smoothing

When action smoothing was increased too much, the hand:

- moved less turbulently
- but then stalled after lift
- or failed to commit to transport

Lesson:

- reducing visible roughness is useful only if task competence is preserved

## Phase 5: Use The Best Historical Runs As Anchors

At this point, two reference behaviors were clear:

- `3rfnpbbk` had the better motion profile before release was solved
- `3hq3kpce` solved the full objective

This led to an important conclusion:

- reward values alone were not enough to combine the advantages
- we needed the ability to initialize from one learned policy and fine-tune toward another behavior

## Phase 6: Add Warm-Start Training

### Change: training can initialize from a checkpoint

We updated the training pipeline to accept an `--init-checkpoint` argument and load an existing SAC model before continuing training.

Reason:

- this allowed us to start from a smoother or more stable earlier policy
- instead of relearning everything from scratch under the newest reward

Effect:

- made it possible to combine:
  - the motion prior from `3rfnpbbk`
  - the release-capable shaping from later models

This was a major workflow improvement.

## Phase 7: Build The Hybrid Motion + Release Model

### Change: hybrid config

We introduced a hybrid training stage:

- start from `3rfnpbbk`
- fine-tune under moderate release-capable shaping

Reason:

- the goal was to preserve the stronger movement quality of the older model
- while adding the release success of the newer one

Effect:

- this produced a better compromise than either model alone
- it became the basis for the final refinement stage

## Phase 8: Finalize The Clean Release Model

### Change: `clean_release`

The final refined config was named `clean_release`.

This stage kept the successful release behavior from the hybrid path, but added a carry stabilization idea:

- after a real lift, and before the hand reaches the goal zone, the transport phase becomes height-stabilized

Specifically:

- transport-phase `z` action is reduced
- a carry-height bonus rewards keeping the object near a target transport height
- a vertical-speed penalty discourages bobbing during transport
- near B, the transport constraint relaxes so descent and release can still happen

Reason:

- the final visible weakness in the best models was turbulent up/down movement while carrying toward B

Effect:

- cleaner carry behavior
- preserved ability to place and release

## Final Working Training Chain

The final working path for the simple hand was:

1. start from the earlier smoother motion model (`3rfnpbbk`)
2. fine-tune with moderate release-capable shaping
3. train the `clean_release` stage
4. continue `clean_release` for further refinement

One important continuation run was:

- `53cisq88`
  - `clean_release_continue`
  - 500000-step continuation

This version was judged to be good enough:

- it completed all objectives
- it still had some roughness
- but the behavior was strong enough to serve as a simple-hand baseline for future work

## Why The Final Version Works Better

The latest usable model works because it addresses the task in the correct order:

1. do not reward fake transport before lift
2. preserve lift/transport value after a valid carry starts
3. reward descent and placement more smoothly near B
4. reward opening and retreating during release
5. penalize clinging and re-contact after release
6. stabilize height during the carry phase so transport is less turbulent

In short, the final progress came from:

- better phase ordering
- better release shaping
- less destructive smoothing
- warm-start training from a stronger earlier motion policy

## Key Lessons Learned

### 1. More training only helps once the task structure is correct

Early in the process, increasing timesteps did not solve core issues because the reward still allowed bad strategies.

Later, once the task sequence was correct, extra training became useful for refinement.

### 2. Release must be rewarded progressively

A binary release signal was too weak. The model needed intermediate reward for:

- being low over B
- opening fingers
- reducing contact
- moving away

### 3. Smoothing is easy to overdo

When smoothing was too strong, the hand became hesitant and transport got worse.

### 4. Successful historical runs should be reused, not discarded

Using `3rfnpbbk` as a warm-start source was more effective than trying to recreate its good motion from scratch.

### 5. Phase-based control is more effective than splitting the full action space

Instead of redesigning the action space into separate lift and move controllers, it was better to:

- keep one action space
- change the reward and action interpretation by phase

## Files Most Relevant To The Final Result

The most important files for the final simple-hand model are:

- `src/contact_aware_rl/config.py`
- `src/contact_aware_rl/env.py`
- `src/contact_aware_rl/experiment.py`
- `src/contact_aware_rl/train.py`
- `tests/test_env.py`
- `configs/cartesian_place_priority_motion_release_hybrid.yaml`
- `configs/clean_release.yaml`

## Conclusion

The final simple-hand model was not achieved by one change. It came from a long trial-and-error process that gradually fixed:

- reward ordering
- transport eligibility
- placement shaping
- release shaping
- post-release behavior
- carry stability
- training workflow through checkpoint initialization

The final result is a practical simple-hand baseline that can:

- grasp
- lift
- transport
- place
- release

with enough consistency to serve as a foundation for future experiments on more complicated hands.
