"""
Picking one milk carton out of a ten-carton clutter with Tracy in MuJoCo, and using the
recorded attempts to compare two causal-query pipelines over the same
``cause``/``causes_effect`` EQL queries: a relational circuit (RSPN grounded into a
``RelationalCausalCircuit``) and a joint probability tree fitted on the same attempts
flattened into one fixed-width table.

The scene, object and attempt classes in :mod:`~experiments.causal_reasoning.tracy_clutter_picking.domain`
follow the shape of the GraspClutter6D dataset (https://sites.google.com/view/graspclutter6d),
so its scenes can replace the ten-milk mock once they are annotated -- see
:mod:`~experiments.causal_reasoning.tracy_clutter_picking.graspclutter6d`.
"""
