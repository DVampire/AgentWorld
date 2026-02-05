#!/bin/bash

# ##########################grok-4.1-fast##########################
# # only prompt optimization
# benchmark=aime24
# model_name=grok-4.1-fast
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime24
# model_name=grok-4.1-fast
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # prompt and solution optimization
# benchmark=aime24
# model_name=grok-4.1-fast
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# benchmark=aime25
# model_name=grok-4.1-fast
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime25
# model_name=grok-4.1-fast
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # prompt and solution optimization
# benchmark=aime25
# model_name=grok-4.1-fast
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 4 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}
# ##########################grok-4.1-fast##########################

# ##########################claude-sonnet-4.5##########################
# # only prompt optimization
# benchmark=aime24
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime24
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # prompt and solution optimization
# benchmark=aime24
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# benchmark=aime25
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime25
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # prompt and solution optimization
# benchmark=aime25
# model_name=claude-sonnet-4.5
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}
# ##########################claude-sonnet-4.5##########################

##########################gpt-4.1##########################
# # only prompt optimization
# benchmark=aime24
# model_name=gpt-4.1
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime24
# model_name=gpt-4.1
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# prompt and solution optimization
# benchmark=aime24
# model_name=gpt-4.1
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# benchmark=aime25
# model_name=gpt-4.1
# optimize_trainable_variables=true   # true or false
# optimize_solution=false              # true or false
# exp_name=prompt
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# # only solution optimization
# benchmark=aime25
# model_name=gpt-4.1
# optimize_trainable_variables=false   # true or false
# optimize_solution=true              # true or false
# exp_name=solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# prompt and solution optimization
# benchmark=aime25
# model_name=gpt-4.1
# optimize_trainable_variables=true   # true or false
# optimize_solution=true              # true or false
# exp_name=prompt_solution
# tag=${model_name}_${benchmark}_${exp_name}_results
# OPT_ARGS=""
# if [ "$optimize_trainable_variables" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_trainable_variables"
# fi
# if [ "$optimize_solution" = "true" ]; then
#     OPT_ARGS="$OPT_ARGS --optimize_solution"
# fi
# python examples/run_tool_calling_agent_experiment_async.py \
#     --config configs/tool_calling_agent.py \
#     --benchmark ${benchmark} \
#     --concurrency 8 \
#     --model_name openrouter/${model_name} \
#     $OPT_ARGS \
#     --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}
##########################gpt-4.1##########################

##########################gpt-4o##########################
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# only solution optimization
benchmark=aime24
model_name=gpt-4o
optimize_trainable_variables=false   # true or false
optimize_solution=true              # true or false
exp_name=solution
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# prompt and solution optimization
benchmark=aime24
model_name=gpt-4o
optimize_trainable_variables=true   # true or false
optimize_solution=true              # true or false
exp_name=prompt_solution
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

benchmark=aime25
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# only solution optimization
benchmark=aime25
model_name=gpt-4o
optimize_trainable_variables=false   # true or false
optimize_solution=true              # true or false
exp_name=solution
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}

# prompt and solution optimization
benchmark=aime25
model_name=gpt-4o
optimize_trainable_variables=true   # true or false
optimize_solution=true              # true or false
exp_name=prompt_solution
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
    --concurrency 8 \
    --model_name openrouter/${model_name} \
    $OPT_ARGS \
    --cfg-options model_name=openrouter/${model_name} workdir=workdir/${tag} tag=${tag} tool_calling_agent.model_name=openrouter/${model_name} tool_calling_agent.workdir=workdir/${tag}
##########################gpt-4o##########################