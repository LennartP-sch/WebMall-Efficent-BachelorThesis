"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""
import os
import logging
from dotenv import load_dotenv
import bgym

from agentlab.agents.visualwebmall_agent.agent import WA_AGENT_4O
from agentlab.agents.webmall_generic_agent import AGENT_4o_VISION, AGENT_4o_MINI, AGENT_4o, AGENT_CLAUDE_SONNET_35, AGENT_37_SONNET
from webmall_overrides.study import WebMallStudy
from pathlib import Path
from datetime import datetime
from analyze_agentlab_results.aggregate_log_statistics import process_study_directory
from analyze_agentlab_results.summarize_study import summarize_all_tasks_in_subdirs


logging.getLogger().setLevel(logging.DEBUG)

from agentlab.agents import dynamic_prompting as dp

from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT

from agentlab.agents.generic_agent.generic_agent import  GenericPromptFlags, GenericAgentArgs
from agentlab.agents.dynamic_prompting import print_final_pruning_stats

FLAGS_default = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=dp.ActionFlags(
        action_set=bgym.HighLevelActionSetArgs(
            subsets=["bid"],
            multiaction=False,
        ),
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=60_000,
    be_cautious=True,
    extra_instructions=None,
    )

FLAGS_AX = FLAGS_default.copy()

#AX_M (STANDARD FLAG)
FLAGS_AX_M = FLAGS_default.copy()
FLAGS_AX_M.use_memory = True

#PRUNNED AX
FLAGS_AX_ADV_M = FLAGS_AX_M.copy()
FLAGS_AX_ADV_M.obs.use_ax_tree_advanced = False
FLAGS_AX_ADV_M.obs.use_ax_tree_amazon = True

#AX-AM (Adjusted Memory)
FLAGS_AX_AM = FLAGS_AX_M.copy()
FLAGS_AX_AM.use_structured_memory = True

#PRUNNED-AX-AM
FLAGS_AX_ADV_AM = FLAGS_AX_ADV_M.copy()
FLAGS_AX_ADV_AM.use_structured_memory = True

#GF-AX (Auxiliary Model Axtree)
FLAGS_AX_LLM_M = FLAGS_AX_M.copy()
FLAGS_AX_LLM_M.obs.use_model_ax_tree = True

#GF-AX-AM (Auxiliary Model Axtree + Adjusted Memory)
FLAGS_AX_LLM_AM = FLAGS_AX_LLM_M.copy()
FLAGS_AX_LLM_AM.use_structured_memory = True

#GF-PRUNNED-AX (Auxiliary Model Axtree)
FLAGS_AX_ADV_LLM_M = FLAGS_AX_M.copy()
FLAGS_AX_ADV_LLM_M.obs.use_amazone_and_model_ax_tree = True

#GF-PRUNNED-AX-AM (Auxiliary Model Axtree + Adjusted Memory)
FLAGS_AX_ADV_LLM_AM = FLAGS_AX_ADV_LLM_M.copy()
FLAGS_AX_ADV_LLM_AM.use_structured_memory = True

#AP (Adjusted Prompt for prefix caching)
FLAGS_AX_M_CACHED = FLAGS_AX_M.copy()
FLAGS_AX_M_CACHED.adjusted_prompt_for_caching = True

#AP + AM (Adjusted Prompt for prefix caching + Adjusted Memory)
FLAGS_AX_AM_CACHED = FLAGS_AX_AM.copy()
FLAGS_AX_AM_CACHED.adjusted_prompt_for_caching_am = True

#AP PRUNNED-AX-AM
FLAGS_AX_ADV_AM_CACHED = FLAGS_AX_ADV_AM.copy()
FLAGS_AX_ADV_AM_CACHED.adjusted_prompt_for_caching_am = True

#AP GF-PRUNNED-AX-AM
FLAGS_AX_ADV_LLM_AM_CACHED = FLAGS_AX_ADV_LLM_AM.copy()
FLAGS_AX_ADV_LLM_AM_CACHED.adjusted_prompt_for_caching_am = True

#AGENTS

## Gemini 2.5 Flash Agents
AGENT_GEMINI_2_5_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_M,
)
AGENT_GEMINI_2_5_AX_M.agent_name = "AGENT_GEMINI_2_5_AX_M"

AGENT_GEMINI_2_5_AX_ADV_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_ADV_M,
)
AGENT_GEMINI_2_5_AX_ADV_M.agent_name = "AGENT_GEMINI_2_5_AX_ADV_M"

AGENT_GEMINI_2_5_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_AM
)
AGENT_GEMINI_2_5_AX_AM.agent_name = "AGENT_GEMINI_2_5_AX_AM"

AGENT_GEMINI_2_5_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_ADV_AM
)
AGENT_GEMINI_2_5_AX_ADV_AM.agent_name = "AGENT_GEMINI_2_5_AX_ADV_AM"

AGENT_GEMINI_2_5_CACHED_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_M_CACHED
)
AGENT_GEMINI_2_5_CACHED_AX_M.agent_name = "AGENT_GEMINI_2_5_CACHED_AX_M"

AGENT_GEMINI_2_5_CACHED_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_AM_CACHED
)
AGENT_GEMINI_2_5_CACHED_AX_AM.agent_name = "AGENT_GEMINI_2_5_CACHED_AX_AM"

AGENT_GEMINI_2_5_CACHED_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-flash"],
    flags=FLAGS_AX_ADV_AM_CACHED
)
AGENT_GEMINI_2_5_CACHED_AX_ADV_AM.agent_name = "AGENT_GEMINI_2_5_CACHED_AX_ADV_AM"


## Gemini 2.5 Pro Agents

AGENT_GEMINI_2_5_PRO_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-pro"],
    flags=FLAGS_AX_M
)
AGENT_GEMINI_2_5_PRO_AX_M.agent_name = "AGENT_GEMINI_2_5_PRO_AX_M"

AGENT_GEMINI_2_5_PRO_AX_M_CACHED = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-pro"],
    flags=FLAGS_AX_M_CACHED
)
AGENT_GEMINI_2_5_PRO_AX_M_CACHED.agent_name = "AGENT_GEMINI_2_5_PRO_AX_M_CACHED"

AGENT_GEMINI_2_5_PRO_LLM_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-pro"],
    flags=FLAGS_AX_LLM_M
)
AGENT_GEMINI_2_5_PRO_LLM_AX_M.agent_name = "AGENT_GEMINI_2_5_PRO_LLM_AX_M"

AGENT_GEMINI_2_5_PRO_AX_ADV_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-pro"],
    flags=FLAGS_AX_ADV_M
)
AGENT_GEMINI_2_5_PRO_AX_ADV_M.agent_name = "AGENT_GEMINI_2_5_PRO_AX_ADV_M"

AGENT_GEMINI_2_5_PRO_AX_ADV_LLM_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["gemini-2.5-pro"],
    flags=FLAGS_AX_ADV_LLM_M
)
AGENT_GEMINI_2_5_PRO_AX_ADV_LLM_M.agent_name = "AGENT_GEMINI_2_5_PRO_AX_ADV_LLM_M"

## Grok 4 Fast (non-reasoning) Agents

AGENT_GROK_4_FAST_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_M,
)
AGENT_GROK_4_FAST_AX_M.agent_name = "AGENT_GROK_4_FAST_AX_M"

AGENT_GROK_4_FAST_AX_ADV_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_ADV_M,
)
AGENT_GROK_4_FAST_AX_ADV_M.agent_name = "AGENT_GROK_4_FAST_AX_ADV_M"

AGENT_GROK_4_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_AM
)
AGENT_GROK_4_AX_AM.agent_name = "AGENT_GROK_4_AX_AM"

AGENT_GROK_4_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_ADV_AM
)
AGENT_GROK_4_AX_ADV_AM.agent_name = "AGENT_GROK_4_AX_ADV_AM"

AGENT_GROK_4_FAST_LLM_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_LLM_M
)
AGENT_GROK_4_FAST_LLM_AX_M.agent_name = "AGENT_GROK_4_FAST_LLM_AX_M"

AGENT_GROK_4_FAST_AX_ADV_LLM_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_ADV_LLM_M
)
AGENT_GROK_4_FAST_AX_ADV_LLM_M.agent_name = "AGENT_GROK_4_FAST_AX_ADV_LLM_M"

AGENT_GROK_4_LLM_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_LLM_AM
)
AGENT_GROK_4_LLM_AX_AM.agent_name = "AGENT_GROK_4_LLM_AX_AM"

AGENT_GROK_4_LLM_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openrouter/x-ai/grok-4-fast"],
    flags=FLAGS_AX_ADV_LLM_AM
)
AGENT_GROK_4_LLM_AX_ADV_AM.agent_name = "AGENT_GROK_4_LLM_AX_ADV_AM"

## GPT-4.1 Agents - Auxiliary Model is gpt-4.1-mini-2025-04-14



AGENT_GPT_41_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_M,
)
AGENT_GPT_41_AX_M.agent_name = "AGENT_GPT_41_AX_M"

AGENT_GPT_41_AX_ADV_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"], 
    flags=FLAGS_AX_ADV_M,
)
AGENT_GPT_41_AX_ADV_M.agent_name = "AGENT_GPT_41_AX_ADV_M"

AGENT_GPT_41_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_AM,
)
AGENT_GPT_41_AX_AM.agent_name = "AGENT_GPT_41_AX_AM"

AGENT_GPT_41_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_ADV_AM,
)
AGENT_GPT_41_AX_ADV_AM.agent_name = "AGENT_GPT_41_AX_ADV_AM"

AGENT_GPT_41_LLM_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_LLM_M,
)
AGENT_GPT_41_LLM_AX_M.agent_name = "AGENT_GPT_41_LLM_AX_M"

AGENT_GPT_41_AX_ADV_LLM_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_ADV_LLM_M,
)
AGENT_GPT_41_AX_ADV_LLM_M.agent_name = "AGENT_GPT_41_AX_ADV_LLM_M"

AGENT_GPT_41_LLM_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_LLM_AM,
)
AGENT_GPT_41_LLM_AX_AM.agent_name = "AGENT_GPT_41_LLM_AX_AM"

AGENT_GPT_41_LLM_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_ADV_LLM_AM,
)
AGENT_GPT_41_LLM_AX_ADV_AM.agent_name = "AGENT_GPT_41_LLM_AX_ADV_AM"

AGENT_GPT_41_CACHED_AX_M = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_M_CACHED,
)
AGENT_GPT_41_CACHED_AX_M.agent_name = "AGENT_GPT_41_CACHED_AX_M"

AGENT_GPT_41_CACHED_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_AM_CACHED,
)
AGENT_GPT_41_CACHED_AX_AM.agent_name = "AGENT_GPT_41_CACHED_AX_AM"

AGENT_GPT_41_CACHED_AX_ADV_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_ADV_AM_CACHED,
)
AGENT_GPT_41_CACHED_AX_ADV_AM.agent_name = "AGENT_GPT_41_CACHED_AX_ADV_AM"

AGENT_GPT_41_CACHED_LLM_AX_AM = GenericAgentArgs(
    chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4.1-2025-04-14"],
    flags=FLAGS_AX_ADV_LLM_AM_CACHED,
)
AGENT_GPT_41_CACHED_LLM_AX_AM.agent_name = "AGENT_GPT_41_CACHED_LLM_AX_AM"


#========================================================================================================================
#   CONFIGUARATION PART START
#========================================================================================================================

# These are the combinations a runed on Gemini 2.5 flash and Grok 4 fast:


## AGENT_GPT_41_AX_M i already run it arleady myself - cost me about 10.64$
## ich denke wenn man alles wie in der liste laufen lässt wird es an die 100$ kosten
## Also wenn das budget dann zu knapp ist, wähle ich noch ein paar aus die weniger priorität haben.


agents_to_run = [

    AGENT_GPT_41_AX_ADV_M,
    AGENT_GPT_41_AX_AM,
    AGENT_GPT_41_AX_ADV_AM,
    AGENT_GPT_41_LLM_AX_M,
    AGENT_GPT_41_AX_ADV_LLM_M,
    AGENT_GPT_41_LLM_AX_AM,
    AGENT_GPT_41_LLM_AX_ADV_AM,
    AGENT_GPT_41_CACHED_AX_M,
    AGENT_GPT_41_CACHED_AX_AM,
    AGENT_GPT_41_CACHED_AX_ADV_AM,

    #Everthing combined:
    AGENT_GPT_41_CACHED_LLM_AX_AM     
]


# choose your agent
# Start with AGENT_GPT_41_AX_ADV_M
agent_args = [AGENT_GPT_41_AX_ADV_M]

#For each agent in agents_to_run, run webmall_short_basic and webmall_short_advanced

benchmark = "test"
#benchmark = "webmall_short_basic"
#benchmark = "webmall_short_advanced"

## Number of parallel jobs
n_jobs = 4 
# n_jobs = -1  # to use all available cores

#========================================================================================================================
#   CONFIGUARATION PART END
#========================================================================================================================


# Set relaunch = True to relaunch an existing study, this will continue incomplete
# experiments and relaunch errored experiments
relaunch = False




# Set reproducibility_mode = True for reproducibility
# this will "ask" agents to be deterministic. Also, it will prevent you from launching if you have
# local changes. For your custom agents you need to implement set_reproducibility_mode
reproducibility_mode = False




if __name__ == "__main__":  # necessary for dask backend

    if reproducibility_mode:
        [a.set_reproducibility_mode() for a in agent_args]

    if relaunch:
        #  relaunch an existing study
        study = WebMallStudy.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = WebMallStudy(agent_args, benchmark, logging_level_stdout=logging.INFO, suffix="AX")
        study_name = study.name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        study_dir = f"{os.getenv('AGENTLAB_EXP_ROOT')}/{timestamp}_{study_name}/"
        study = WebMallStudy(agent_args, benchmark, logging_level_stdout=logging.INFO, dir=study_dir, suffix="AX")
        
    parallel_backends = ["sequential", "ray"]
    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backends[1],
        strict_reproducibility=reproducibility_mode,
        n_relaunch=1,
    )

    if reproducibility_mode:
        study.append_to_journal(strict_reproducibility=True)

    print("\n" + "=" * 80)
    print("📊 Aggregating Auxiliary Model Pruning Statistics...")
    print("=" * 80)

    # ✅ Lese Stats aus temporärer Datei
    import json
    import os
    
    stats_file = os.path.join(os.getenv('AGENTLAB_EXP_ROOT', '.'), 'aux_pruning_stats_temp.json')
    
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            all_worker_stats = json.load(f)
        
        # ✅ Aggregiere Stats von allen Ray Workers (SUMMIERE über alle Workers!)
        total_calls = 0
        total_input = 0
        total_output = 0
        total_cached_tokens = 0
        
        for worker_id, stats in all_worker_stats.items():
            total_calls += stats['call_count']
            total_input += stats['total_input_tokens']
            total_output += stats['total_output_tokens']
            total_cached_tokens += stats.get('total_cached_tokens', 0)  # ✅ Cached Tokens aggregieren
        
        if total_calls > 0:
            total_tokens = total_input + total_output
            avg_input_percall = total_input / total_calls
            avg_output_percall = total_output / total_calls
            avg_cached_percall = total_cached_tokens / total_calls if total_cached_tokens > 0 else 0
            
            print(f"\n   Total API Calls: {total_calls}")
            print(f"   Input Tokens: {total_input:,}")
            print(f"   Output Tokens: {total_output:,}")
            print(f"   Cached Tokens: {total_cached_tokens:,}")  # ✅ Cached Tokens ausgeben
            print(f"   Total Tokens: {total_tokens:,}")
            print(f"   Avg Input per Call: {avg_input_percall:.2f}")
            print(f"   Avg Output per Call: {avg_output_percall:.2f}")
            print(f"   Avg Cached per Call: {avg_cached_percall:.2f}")  # ✅ Cached Tokens Durchschnitt
            print("=" * 80 + "\n")
            
            # ✅ Speichere finale Stats (inkl. Cached Tokens)
            final_stats = {
                "call_count": total_calls,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_cached_tokens": total_cached_tokens,  # ✅ NEU
                "total_tokens": total_tokens,
                "avg_input_tokens": round(avg_input_percall, 2),
                "avg_output_tokens": round(avg_output_percall, 2),
                "avg_cached_tokens": round(avg_cached_percall, 2)  # ✅ NEU
            }
            
            final_stats_file = os.path.join(study_dir, "aux_pruning_stats.json")
            with open(final_stats_file, "w") as f:
                json.dump(final_stats, f, indent=2)
            print(f"✅ Token statistics saved to: {final_stats_file}\n")
            
            # ✅ Cleanup: Lösche temp Datei
            os.remove(stats_file)
        else:
            print("⚠️ No pruning statistics found\n")
    else:
        print("⚠️ No pruning statistics file found (auxiliary model pruning was not used)\n")

    summarize_all_tasks_in_subdirs(study_dir)
    process_study_directory(study_dir)