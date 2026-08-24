---
name: plan-with-theory
trigger: the world model verified at accuracy >= 0.6 and no plan was searched for 3 turns
placeholders: acc
---
[skill: plan-with-theory] Your theory is VERIFIED (accuracy {acc}). Stop probing one action at a
time and search with it instead.

Write goal(grid) -> bool for the state you want -- a colour gone, a piece on its target square,
the HUD counter advanced -- then:

    res = plan_with_theory(predict, goal)
    if res['plan']:
        action(res['plan'])

The search explores the states your own predict() forecasts, so it costs ZERO real actions, and
the plan comes back as specs action() takes verbatim.

Options: actions=[...] chooses the branching set (defaults to the directional and interact
actions; pass explicit CLICK dicts like {{'action':'CLICK','x':12,'y':30}} to search over chosen
targets), max_depth (6), max_nodes (1500).

If it finds nothing, res['reason'] tells you which it is: the budget ran out (loosen goal() or
raise max_depth), the goal is unreachable under your theory (the theory is missing the mechanic
that matters), or the theory is not accurate enough to plan with at all.

If res['note'] is set (plans with more than one step), verify_theory only checked single
already-observed transitions -- steps 2+ predict from states nothing ever actually visited, so a
long plan can still fail mid-way on a mechanic that saturates or changes after a few moves even at
high verified_accuracy. For longer plans, consider running it in smaller chunks and checking
board_changed between them instead of firing every step in one action(res['plan']) call.

Why this beats probing: the score divides the human action count by yours and squares it, so a
searched plan is worth far more than the same moves discovered by trial and error.
