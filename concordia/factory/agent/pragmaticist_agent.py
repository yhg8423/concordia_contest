# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib
from concordia.typing import entity as entity_lib

class SituationAnalysis(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    """This component analyzes the agent's current situation."""

    def __init__(self, **kwargs):
        super().__init__(
            question=(
                "Please analyze {agent_name}'s current situation. "
                "Provide a comprehensive assessment considering the surrounding environment, "
                "relationships with other agents, current tasks or challenges, and any other relevant factors."
                "Please answer in the format `{agent_name} thinks that the surrounding environment is ..., relationships with other agents are ..., current tasks or challenges are ..., and any other relevant factors are ...`"
            ),
            answer_prefix="{agent_name} thinks that ",
            add_to_memory=False,
            memory_tag='[situation_analysis]',
            **kwargs,
        )

class PragmaticReflectiveEvaluation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    """This component reflectively evaluates the agent's previous actions and decisions from a pragmatic perspective."""

    def __init__(self, **kwargs):
        super().__init__(
            question=(
                "Please reflectively evaluate {agent_name}'s previous action and decision from a pragmatic perspective. "
                "Analyze the following aspects:\n"
                "1. Efficiency: How effective were they in achieving goals?\n"
                "2. Practicality: Were the decisions realistic and feasible?\n"
                "3. Cost-effectiveness: Were the results appropriate for the resources invested?\n"
                "4. Adaptability: How well did they respond to changing situations?\n"
                "5. Room for improvement: How can more practical decisions be made in the future?\n"
                "Provide a brief explanation for each item and give an overall pragmatic assessment on a scale of 1-10."
                "Please answer in the format `{agent_name} evaluates as the previous action and decision is ..., because ...`"
            ),
            answer_prefix="{agent_name} evaluates as the previous action and decision is ",
            add_to_memory=True,
            memory_tag='[pragmatic_reflective_evaluation]',
            **kwargs,
        )

class PragmaticOptionPerception(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    """This component considers current action options and alternatives based on previous action reflection and pragmatic view."""

    def __init__(self, **kwargs):
        super().__init__(
            question=(
                "Based on the reflection of previous actions and a pragmatic perspective, please provide the options and alternatives available in {agent_name}'s current situation. "
                "Include the following in your analysis:\n"
                "1. Available options: Feasible actions that can be taken in the current situation\n"
                "2. Practicality of each option: How realistic and implementable each option is\n"
                "3. Expected outcomes: Short-term and long-term results anticipated for each option\n"
                "4. Alternative approaches: Creative alternatives to consider beyond the main options\n"
                "5. Recommended action: The most practical and effective option or alternative\n"
                "Provide a brief explanation for each item and rate the practicality of each option on a scale of 1-10."
                f"The options are {entity_lib.ActionSpec.options}"
                "Please answer in the format: `The available options for {agent_name} are ..., the practicality of each option is ..., the expected outcomes are ..., the alternative approaches are ..., and the recommended action is ...`"
            ),
            answer_prefix="The available options for {agent_name} are ",
            add_to_memory=False,
            memory_tag='[pragmatic_option_perception]',
            **kwargs,
        )

class PragmaticOptionSelection(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
    """This component ranks the given options in order of priority (1st, 2nd, 3rd) based on pragmatic considerations."""

    def __init__(self, **kwargs):
        super().__init__(
            question=(
                "Based on the pragmatic option analysis and the principle of finding practical alternatives in complex situations, "
                "rank the top 2 options for {agent_name} in order of priority (1st, 2nd). Consider the following:\n"
                "1. Feasibility: How realistic and implementable is each option given current constraints?\n"
                "2. Compromise: Which options offer the best balance between ideal outcomes and practical limitations?\n"
                "3. Effectiveness: Which options are most likely to achieve the desired results, even if not perfect?\n"
                "4. Adaptability: Which options allow for flexibility and adjustment as circumstances change?\n"
                "5. Resource efficiency: Which options make the best use of available resources?\n"
                "Provide a brief explanation for each ranking, highlighting how it represents a good solution "
                "given real-world constraints and complexities. Rate the pragmatic value of each chosen option on a scale of 1-10.\n"
                "Please answer in the format: `{agent_name}'s pragmatic option rankings are 1st ... and 2nd ... because ... The pragmatic value of this options are .../10 and .../10.`"
            ),
            answer_prefix=f"{{agent_name}}'s pragmatic option rankings are ",
            add_to_memory=False,
            memory_tag='[pragmatic_option_selection]',
            **kwargs,
        )


class ActionEmphasis(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component represents the agent's action emphasis."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "What is the action that {agent_name} has decided to take? "
            "Is this action currently executable? "
            "If not, what is the next alternative action?"
            "Please answer in the format: `{agent_name} has decided to ... This action is (executable/not executable) ... The alternative action is .... Therefore {agent_name} will ...`"
        ),
        answer_prefix="{agent_name} has decided to ",
        add_to_memory=False,
        memory_tag='[action emphasis]',
        **kwargs,
    )


def _get_class_name(object_: object) -> str:
  return object_.__class__.__name__


def build_agent(
    *,
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> entity_agent_with_logging.EntityAgentWithLogging:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  del update_time_interval
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  raw_memory = legacy_associative_memory.AssociativeMemoryBank(memory)

  measurements = measurements_lib.Measurements()
  instructions = agent_components.instructions.Instructions(
      agent_name=agent_name,
      logging_channel=measurements.get_channel('Instructions').on_next,
  )

  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = 'Summary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )

  relevant_memories_label = '\nRecalled memories and observations'
  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      model=model,
      components={
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(time_display): 'The current date/time is'},
      num_memories_to_retrieve=10,
      pre_act_key=relevant_memories_label,
      logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
  )

  pragmaticist_label = f'\n{agent_name}\'s Pragmaticist Behavior '
  pragmaticist = agent_components.constant.Constant(
    state=(f'{agent_name} is a pragmatist. '
           f'{agent_name} consistently chooses actions that lead to the most practical and useful results. '
           f'While valuing efficiency and effectiveness, {agent_name} also considers how decisions impact broader contexts and real-world situations. '
           f'{agent_name} thrives on evaluating options through a lens of practicality, seeking to maximize both personal and societal benefits. '
           f'{agent_name} view every decision as a chance to solve problems and create tangible improvements. '
           f'In any given scenario, {agent_name} is quick to identify actionable paths that offer the best chance of success, '
           f'leveraging both past experience and empirical evidence to guide their choices. '
           f'When making decisions, {agent_name} avoids abstract theories in favor of solutions that deliver immediate and measurable outcomes. '
           f'Flexibility is a key trait of {agent_name}, allowing {agent_name} to adapt strategies based on new information or shifting circumstances. '
           f'{agent_name} is willing to revise plans if it means achieving better results, showing a preference for iterative problem-solving. '
           f'In negotiations or discussions, {agent_name} is driven by a desire to find the most practical and mutually beneficial agreements. '
           f'{agent_name} tends to be open to compromise, as long as it advances the overall goal of producing effective results. '
           f'{agent_name} often rationalizes their approach by emphasizing the importance of "getting things done" and "focusing on what works," '
           f'while also highlighting the need to address real-world challenges and provide value to others. '
           f'This outlook allows {agent_name} to describe {agent_name} as "results-oriented," "adaptive," and "goal-driven," '
           f'believing that by prioritizing effectiveness and utility, {agent_name} can contribute to both personal success and collective progress.'),
    pre_act_key=pragmaticist_label,
    logging_channel=measurements.get_channel('Pragmaticist').on_next)

  situation_analysis_label = '\nSituation analysis'
  situation_analysis = SituationAnalysis(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        pragmaticist_label: pragmaticist_label,
    },
    clock_now=clock.now,
    pre_act_key=situation_analysis_label,
    logging_channel=measurements.get_channel('SituationAnalysis').on_next,
  )

  pragmatic_reflective_evaluation_label = (
      f'\nQuestion: Please reflectively evaluate {agent_name}\'s previous action and decision from a pragmatic perspective. '
      f'Analyze the following aspects:\n'
      f'1. Efficiency: How effective were they in achieving goals?\n'
      f'2. Practicality: Were the decisions realistic and feasible?\n'
      f'3. Cost-effectiveness: Were the results appropriate for the resources invested?\n'
      f'4. Adaptability: How well did they respond to changing situations?\n'
      f'5. Room for improvement: How can more practical decisions be made in the future?\n'
      f'Provide a brief explanation for each item and give an overall pragmatic assessment on a scale of 1-10.\n'
      f'Please answer in the format `{agent_name} evaluates as the previous action and decision is ..., because ...`'
      '\nAnswer: '
  )
  pragmatic_reflective_evaluation = PragmaticReflectiveEvaluation(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        pragmaticist_label: pragmaticist_label,
        _get_class_name(situation_analysis): situation_analysis_label,
    },
    clock_now=clock.now,
    pre_act_key=pragmatic_reflective_evaluation_label,
    logging_channel=measurements.get_channel('PragmaticReflectiveEvaluation').on_next,
  )

  pragmatic_option_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    pragmatic_option_perception_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  pragmatic_option_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      pragmaticist_label: pragmaticist_label,
      _get_class_name(situation_analysis): situation_analysis_label,
      _get_class_name(pragmatic_reflective_evaluation): pragmatic_reflective_evaluation_label,
  })
  pragmatic_option_perception_label = (
      f'\nQuestion: Based on the reflection of previous actions and a pragmatic perspective, please provide the options and alternatives available in {agent_name}\'s current situation.'
      f'Include the following in your analysis:\n'
      f'1. Available options: Feasible actions that can be taken in the current situation\n'
      f'2. Practicality of each option: How realistic and implementable each option is\n'
      f'3. Expected outcomes: Short-term and long-term results anticipated for each option\n'
      f'4. Alternative approaches: Creative alternatives to consider beyond the main options\n'
      f'5. Recommended action: The most practical and effective option or alternative\n'
      f'Provide a brief explanation for each item and rate the practicality of each option on a scale of 1-10.\n'
      f'The options are {entity_lib.ActionSpec.options}\n'
      f'Please answer in the format: `The available options for {agent_name} are ..., the practicality of each option is ..., the expected outcomes are ..., the alternative approaches are ..., and the recommended action is ...`'
      f'\nAnswer')
  pragmatic_option_perception = (
      PragmaticOptionPerception(
          model=model,
          components=pragmatic_option_perception_components,
          clock_now=clock.now,
          pre_act_key=pragmatic_option_perception_label,
          logging_channel=measurements.get_channel(
              'PragmaticOptionPerception'
          ).on_next,
      )
  )
  # options_perception_components = {}
  # if config.goal:
  #   goal_label = '\nOverarching goal'
  #   overarching_goal = agent_components.constant.Constant(
  #       state=config.goal,
  #       pre_act_key=goal_label,
  #       logging_channel=measurements.get_channel(goal_label).on_next)
  #   options_perception_components[goal_label] = goal_label
  # else:
  #   goal_label = None
  #   overarching_goal = None

  # options_perception_components.update({
  #     _get_class_name(observation): observation_label,
  #     _get_class_name(observation_summary): observation_summary_label,
  #     _get_class_name(relevant_memories): relevant_memories_label,
  #     pragmaticist_label: pragmaticist_label,
  #     _get_class_name(situation_analysis): situation_analysis_label,
  #     _get_class_name(pragmatic_reflective_evaluation): pragmatic_reflective_evaluation_label,
  # })
  # options_perception_label = (
  #     f'\nQuestion: Which options are available to {agent_name} '
  #     'right now?\nAnswer')
  # options_perception = (
  #     agent_components.question_of_recent_memories.AvailableOptionsPerception(
  #         model=model,
  #         components=options_perception_components,
  #         clock_now=clock.now,
  #         pre_act_key=options_perception_label,
  #         logging_channel=measurements.get_channel(
  #             'AvailableOptionsPerception'
  #         ).on_next,
  #     )
  # )

  pragmatic_option_selection_label = (
      f'\nQuestion: Based on the pragmatic option analysis and the principle of finding practical alternatives in complex situations, '
      f'rank the top 2 options for {agent_name} in order of priority (1st, 2nd). Consider the following:\n'
      f'1. Feasibility: How realistic and implementable is each option given current constraints?\n'
      f'2. Compromise: Which options offer the best balance between ideal outcomes and practical limitations?\n'
      f'3. Effectiveness: Which options are most likely to achieve the desired results, even if not perfect?\n'
      f'4. Adaptability: Which options allow for flexibility and adjustment as circumstances change?\n'
      f'5. Resource efficiency: Which options make the best use of available resources?\n'
      f'Provide a brief explanation for each ranking, highlighting how it represents a good solution '
      f'given real-world constraints and complexities. Rate the pragmatic value of each chosen option on a scale of 1-10.\n'
      f'Please answer in the format: `{agent_name}\'s pragmatic option rankings are 1st ... and 2nd ... because ... The pragmatic value of this options are .../10 and .../10.`'
      '\nAnswer'
  )
  pragmatic_option_selection = {}
  if config.goal:
    pragmatic_option_selection[goal_label] = goal_label
  pragmatic_option_selection.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      pragmaticist_label: pragmaticist_label,
      _get_class_name(situation_analysis): situation_analysis_label,
      _get_class_name(pragmatic_reflective_evaluation): pragmatic_reflective_evaluation_label,
      _get_class_name(pragmatic_option_perception): pragmatic_option_perception_label,
  })
  pragmatic_option_selection = (
      PragmaticOptionSelection(
          model=model,
          components=pragmatic_option_selection,
          clock_now=clock.now,
          pre_act_key=pragmatic_option_selection_label,
          logging_channel=measurements.get_channel(
              'PragmaticOptionSelection'
          ).on_next,
      )
  )

  action_emphasis_label = f'\nQuestion: What is the action that {agent_name} has decided to take? Is this action currently executable? If not, what is the next alternative action?\nAnswer'
  action_emphasis = ActionEmphasis(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        pragmaticist_label: pragmaticist_label,
        _get_class_name(situation_analysis): situation_analysis_label,
        _get_class_name(pragmatic_reflective_evaluation): pragmatic_reflective_evaluation_label,
        _get_class_name(pragmatic_option_perception): pragmatic_option_perception_label,
        _get_class_name(pragmatic_option_selection): pragmatic_option_selection_label,
    },
    clock_now=clock.now,
    pre_act_key=action_emphasis_label,
    logging_channel=measurements.get_channel('ActionEmphasis').on_next,
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      situation_analysis,
      pragmatic_reflective_evaluation,
      pragmatic_option_perception,
      pragmatic_option_selection,
      action_emphasis,
  )
  components_of_agent = {_get_class_name(component): component
                         for component in entity_components}
  components_of_agent[
      agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME] = (
          agent_components.memory_component.MemoryComponent(raw_memory))

  component_order = list(components_of_agent.keys())
  if overarching_goal is not None:
    components_of_agent[goal_label] = overarching_goal
    # Place goal after the instructions.
    component_order.insert(1, goal_label)

  components_of_agent[pragmaticist_label] = pragmaticist
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    pragmaticist_label)

  act_component = agent_components.concat_act_component.ConcatActComponent(
      model=model,
      clock=clock,
      component_order=component_order,
      logging_channel=measurements.get_channel('ActComponent').on_next,
  )

  agent = entity_agent_with_logging.EntityAgentWithLogging(
      agent_name=agent_name,
      act_component=act_component,
      context_components=components_of_agent,
      component_logging=measurements,
  )

  return agent
