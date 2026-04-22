high_prompt = """You are a high-level planner. Based on the state (task description, group action and current observation), please generate a clear and simple subtask.\n"""
low_prompt = """You are a low-level action executor. Based on the current subtask and observation, please generate a executable action and determine if the subtask is completed (true/false).\n 
Sometimes the number you receive will be wrong.
Your task is to infer the correct number from the observation. \n """
subtask_complete_prompt = """Determine if the low-level actions successfully completed the given subtask by high-level:
Subtask: [subtask]
Initial observation: [initial_obs]
Actions: [action_sequence]
Final observation: [final_obs]
Output only a single digit:
"True" - if the actions successfully completed the subtask
"False" - if the actions failed to complete the subtask
Give the "True" or "False": """


textcraft_high_prompt = (
  "You are a high-level planner. Based on the state (task description, group action and current observation), "
  "please generate a clear and simple subtask.\n"
  "Task Description:\n"
)

textcraft_low_prompt = (
  "You are a low-level action executor. Based on the current subtask and observation, "
  "please generate an executable action and determine if the subtask is completed (true/false).\n"
  "Subtask: "
)

textcraft_obs_prefix = "Current observation: "
single_prompt = """You are an action executor. Based on the state (task description, group action and current observation), please generate a executable action and determine if the task is completed (true/false).\n """