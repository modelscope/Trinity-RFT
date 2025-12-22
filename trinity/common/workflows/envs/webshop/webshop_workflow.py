# -*- coding: utf-8 -*-
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task

SPARSE_REWARD = False

EXAMPLE_PROMPT = """
Observation:
Webshop
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[button] Search [button_]

Response:
<think>OK, let's search for 3 ounce bright citrus deodorant sensitive skin</think><action>search[3 ounce bright citrus deodorant sensitive skin]</action>

Observation:
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[button] Back to Search [button_]
Page 1 (Total results: 50)
[button] Next > [button_]
[button] B078GWRC1J [button_]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[button] B078GTKVXY [button_]
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
$10.99
[button] B08KBVJ4XN [button_]
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack)
$15.95

Response:
<think>button B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first</think><action>click[B078GWRC1J]</action>

Observation:
Instruction:
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars
[button] Back to Search [button_]
[button] < Prev [button_]
scent [button] assorted scents [button_][button] bright citrus [button_][button] calming lavender [button_][button] ginger fresh [button_][button] simply non-scents [button_]
size [button] travel set (4-pack) [button_][button] 3 ounce (pack of 1) [button_][button] 3-ounce (2-pack) [button_]
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce
Price: $10.99
Rating: N.A.
[button] Description [button_]
[button] Features [button_]
[button] Reviews [button_]
[button] Buy Now [button_]

Response:
<think>For 3 ounce bottle of bright citrus deodorant for sensitive skin, the item has options 'bright citrus' and '3 ounce (pack of 1)' and seems good to buy. </think><action>click[bright citrus]</action>

Observation:
You have clicked bright citrus.
...

Response:
<think>Now I should select the 3 ounce (pack of 1) option</think><action>click[3 ounce (pack of 1)]</action>

Observation:
You have clicked 3 ounce (pack of 1).
...

Response:
<think>I can buy the item</think><action>click[Buy Now]</action>
"""


WebShop_SYSTEM_PROMPT_WITH_EXAMPLE = f"""
You are an agent interacting with a virtual text-based web shopping environment to test out your ability. Your job is to follow the Instruction provided and mimic the steps to buy the item that are closest to the Instruct provided.

## Action Format:
You should give both the action_name and action_arg like the format `action_name[action_arg]`. You can execute two types of actions, search and click.
- When the button `[button] Search [button_]` is available in the current observation, you can execute the action <action>search[xxx]</action> (you should type the query you want to search in the square brackets here).
- You can click buttons `[button] xxx [button_]` that is available in the current observation, by execute the action <action>click[xxx]</action>.

Below are some examples of action formats.
- <action>search[white shoes]</action>
- <action>click[Buy Now]</action>

## Example:
Here is an example:
```
{EXAMPLE_PROMPT}
```

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
You should finish the task and buy the item within 15 steps.
DO NOT try to interact with the user at anytime. Finish the task and buy the item by yourself.
"""

WebShop_SYSTEM_PROMPT = """
You are an agent interacting with a virtual text-based web shopping environment to test out your ability. Your job is to follow the Instruction provided and mimic the steps to buy the item that are closest to the Instruction provided.

## Action Format:
You should give both the action_name and action_arg like the format `action_name[action_arg]`. You can execute two types of actions, search and click.
- When the button `[button] Search [button_]` is available in the current observation, you can execute the action <action>search[xxx]</action> (you should type the query you want to search in the square brackets here).
- You can click buttons `[button] xxx [button_]` that is available in the current observation, by execute the action <action>click[xxx]</action>.

Below are some examples of action formats.
- <action>search[white shoes]</action>
- <action>click[Buy Now]</action>

## Notes:
At each step, you should first think then perform action to fulfill the instruction. You should ALWAYS wrap your thinking with the <think> </think> tag and wrap your action with the <action> </action> tag.
You should ALWAYS take one action each step.
You should finish the task and buy the item within 15 steps.
DO NOT try to interact with the user at anytime. Finish the task and buy the item by yourself.
"""


def format_observation(observation: str):
    return "Observation: " + observation


def parse_action(response):
    try:
        # parse the action within the <action> </action> tag
        action = response.split("<action>")[1].split("</action>")[0].strip()
        return action
    except Exception as e:
        print("Error parsing action:", e)
        return ""


def validate_action(action, available_actions):
    # parse action name and args
    import re

    pattern = re.compile(r"(.+)\[(.+)\]")
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()

    if action_arg is not None:
        action_arg = action_arg.lower()
    # for correct action format
    if (
        action_name == "search"
        and action_arg is not None
        and action_arg != ""
        and available_actions["has_search_bar"]
    ):
        return True, ""
    elif action_name == "click" and action_arg in available_actions["clickables"]:
        return True, ""
    # for incorrect action format
    if action_name == "search":
        if action_arg == "" or action_arg is None:
            return (
                False,
                "Invalid action, please type in the query you want to search in the square brackets here.",
            )
        else:
            return (
                False,
                "Can not perform search action without search bar. Please click the Back to Search button first.",
            )
    elif action_name == "click":
        if action_arg not in available_actions["clickables"]:
            return (
                False,
                f"Incorrect action format, make sure you have the correct button name that is **within the current page**. The buttons you can click now are: {available_actions['clickables']}",
            )
    return (
        False,
        "Invalid action. You should wrap your action with the <action> </action> tag and follow the action format.",
    )


class WebShopWorkflow(MultiTurnWorkflow):
    """A workflow for webshop task."""

    can_reset: bool = True
    is_async: bool = True

    def __init__(
        self,
        model: ModelWrapper,
        task: Task,
        auxiliary_models: Optional[List] = None,
    ):
        super().__init__(
            model=model,
            task=task,
        )
        self.max_env_steps = 15
        self.reset(task)

        # TODO: Make parallel envs
        try:
            import gym
            from web_agent_site.envs import WebAgentTextEnv  # noqa: F401
        except Exception as e:
            print("Please make sure you have installed the web_agent_site package.")
            error_message = f"Error importing WebAgentTextEnv {str(e)}. Please make sure you have installed the web_agent_site package, following the instructions in https://github.com/princeton-nlp/WebShop"
            raise ImportError(error_message)
        print("Making GYM env")
        # NOTE: Hosting the env require ~15GB CPU memory.
        # If you want easier env, you can set the num_products to 1000 or 100000.
        self.env = gym.make(
            "WebAgentTextEnv-v0", observation_mode="text_rich", num_products=None, human_goals=True
        )

    def reset(self, task: Task):
        self.task_desc = task.task_desc or "0"
        self.repeat_times = task.repeat_times

    async def get_model_response(self, messages):
        responses = await self.model.chat_async(messages, n=1)
        return responses

    async def get_model_response_text(self, messages):
        return (await self.get_model_response(messages))[0].response_text

    async def generate_env_inference_samples(
        self, env, session_id, rollout_num
    ) -> List[Experience]:
        # TODO: Make this parallel
        print("Generating env inference samples...")
        experience_list = []
        for i in range(rollout_num):
            env.reset(session=session_id)
            final_reward = -0.1
            observation = env.observation
            memory = []
            memory.append({"role": "system", "content": WebShop_SYSTEM_PROMPT})
            for r in range(self.max_env_steps):
                available_actions = env.get_available_actions()
                format_obs = format_observation(observation)
                memory = memory + [{"role": "user", "content": format_obs}]
                response_text = await self.get_model_response_text(memory)
                memory.append({"role": "assistant", "content": response_text})
                action = parse_action(response_text)
                action_valid, error_msg = validate_action(action, available_actions)
                if action_valid:
                    observation, reward, done, info = env.step(action)
                else:
                    observation = error_msg
                    reward = 0
                    done = False
                if done:
                    final_reward = reward
                    break
            if SPARSE_REWARD:
                if final_reward >= 0.99:
                    final_reward = 1
                elif final_reward >= 0:
                    final_reward = 0
                else:
                    final_reward = -0.1
            experience = self.process_messages_to_experience(
                memory,
                final_reward,
                {"session_id": session_id, "env_rounds": r, "env_done": 1 if done else 0},
            )
            experience_list.append(experience)
        return experience_list

    async def run_async(self) -> List[Experience]:
        # assume the task_description is the session_id generated.
        session_id = int(self.task_desc)
        rollout_n = self.repeat_times
        return await self.generate_env_inference_samples(self.env, session_id, rollout_n)
