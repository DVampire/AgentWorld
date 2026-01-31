#!/bin/bash

# only prompt optimization
benchmark=aime24
model_name=gpt-4o
optimize_trainable_variables=true   # true or false
optimize_solution=false              # true or false
exp_name=prompt
tag=${model_name}_${benchmark}_${exp_name}_results
OPT_ARGS=""
if [ "$optimize_trainable_variables" = "true" ]; then
    OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
fi
if [ "$optimize_solution" = "true" ]; then
    OPT_ARGS="$OPT_ARGS --optimize_solution"
fi
python examples/run_tool_calling_agent_experiment_async.py \
    --config configs/tool_calling_agent.py \
    --benchmark ${benchmark} \
    --concurrency 4 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}
