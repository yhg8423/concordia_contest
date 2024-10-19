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


class SituationAssessment(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component evaluates the importance of social value and personal benefit in the current situation."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Analyze the current situation {agent_name} is facing and evaluate the relative importance "
            "of social value and personal benefit. Consider the following factors: Context and scope of the situation, Short-term and long-term consequences, Ethical aspects, Potential impact on society and the individual\n"
            "After your assessment, please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks social value importance is 3 and personal benefit importance is 7, because ...`,"
            "`{agent_name} thinks social value importance is 6 and personal benefit importance is 4, because ...`"
            "\nHigher scores indicate greater importance of that aspect. Please express your evaluation on a scale from 0 to 10."
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=True,
        memory_tag='[situation importance assessment]',
        **kwargs,
    )

class ActionEmphasis(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component represents the agent's action emphasis."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "What is the action that {agent_name} has decided to take?"
        ),
        answer_prefix="{agent_name} has decided to ",
        add_to_memory=False,
        memory_tag='[action emphasis]',
        **kwargs,
    )


class SocialValueEvaluation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component evaluates the social value of each option."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "For each option {agent_name} is considering, evaluate its social value "
            "on a scale of 0 to 10. Social value is defined as the potential benefit "
            "to others, including but not limited to: improving others' well-being, "
            "promoting fairness and equality, fostering community growth, and "
            "contributing to long-term societal progress. Provide a score and a brief "
            "explanation for each option. please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks social value is 4, because ...`, `{agent_name} thinks social value is 7, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=False,
        memory_tag='[social value evaluation]',
        **kwargs,
    )

class PersonalBenefitEvaluation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component evaluates the personal benefit of each option."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "For each option {agent_name} is considering, evaluate its personal benefit "
            "on a scale of 0 to 10. Personal benefit is defined as the potential gain "
            "for oneself, including but not limited to: improving one's own well-being, "
            "promoting personal success and achievement, fostering individual growth, and "
            "contributing to long-term personal progress. Provide a score and a brief "
            "explanation for each option. please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=False,
        memory_tag='[personal benefit evaluation]',
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

  social_opportunist_label = f'\n{agent_name}\'s Social Opportunist Behavior '
  social_opportunist = agent_components.constant.Constant(
    state=(f'{agent_name} is extremely opportunistic and always agrees to '
           'any proposition that offers even the slightest personal benefit. '
           f'However, {agent_name} also considers the benefits to others and society as a whole. '
           f'{agent_name} is hyper-focused on potential gains, both personal and collective, '
           'and seeks to maximize the total benefit.'
           f'{agent_name} views every interaction as a potential opportunity '
           f'for personal gain and societal improvement. {agent_name} is quick to spot advantages in any '
           'situation and eagerly pursues them, while also considering how these opportunities '
           'can benefit others.'
           f'In decision-making, {agent_name} prioritizes options that offer the greatest '
           f'combined benefit to self, others, and society. {agent_name} has a strong bias '
           'towards action, preferring to seize opportunities that create value for all parties.'
           f'{agent_name} tends to carefully evaluate potential gains and losses, both personal '
           f'and collective. {agent_name} is willing to take calculated risks if there\'s a possibility '
           'of significant overall benefit. This can lead to innovative solutions that '
           'create win-win situations.'
           f'In negotiations, {agent_name} is easily swayed by offers that include personal benefits, '
           f'but also considers how these offers impact others and society. {agent_name} may agree '
           f'to terms that balance personal gain with broader positive outcomes. {agent_name}\'s '
           f'eagerness to benefit extends beyond self-interest to encompass collective welfare.'
           f'To rationalize {agent_name}\'s behavior, {agent_name} often emphasizes the '
           'importance of "seizing opportunities" and "not letting chances slip '
           f'by", while also highlighting the value of "social responsibility" and "mutual benefit". '
           f'{agent_name} might describe {agent_name}self as "proactive", "ambitious", and "socially conscious".'
           f'{agent_name} believes this approach will help maximize personal gains '
           'while also contributing positively to others and society as a whole. {agent_name} aims to '
           'make decisions that create the greatest total value, considering both individual and '
           'collective interests.'),
    pre_act_key=social_opportunist_label,
    logging_channel=measurements.get_channel('SocialOpportunist').on_next)

  options_perception_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    options_perception_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  options_perception_components.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      social_opportunist_label: social_opportunist_label,
  })
  options_perception_label = (
      f'\nQuestion: Which options are available to {agent_name} '
      'right now?\nAnswer')
  options_perception = (
      agent_components.question_of_recent_memories.AvailableOptionsPerception(
          model=model,
          components=options_perception_components,
          clock_now=clock.now,
          pre_act_key=options_perception_label,
          logging_channel=measurements.get_channel(
              'AvailableOptionsPerception'
          ).on_next,
      )
  )

  social_value_evaluation_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate its social value '
      f'on a scale of 0 to 10. Social value is defined as the potential benefit '
      f'to others, including but not limited to: improving others\' well-being, '
      f'promoting fairness and equality, fostering community growth, and '
      f'contributing to long-term societal progress. Provide a score and a brief '
      f'explanation for each option. please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks social value is 4, because ...`, `{agent_name} thinks social value is 7, because ...`'
      '\nAnswer'
  )
  social_value_evaluation = SocialValueEvaluation(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    pre_act_key=social_value_evaluation_label,
    logging_channel=measurements.get_channel('SocialValueEvaluation').on_next,
  )

  personal_benefit_evaluation_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate its personal benefit '
      f'on a scale of 0 to 10. Personal benefit is defined as potential advantages to {agent_name}, '
      f'including but not limited to: improving their well-being, financial gains, enhancing social status, '
      f'personal growth, and enjoyment. Provide a score and a brief explanation for each option. please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`'
      f'\nAnswer'
  )

  personal_benefit_evaluation = PersonalBenefitEvaluation(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    pre_act_key=personal_benefit_evaluation_label,
    logging_channel=measurements.get_channel('PersonalBenefitEvaluation').on_next,
  )

  situation_assessment_label = (
      f'\nQuestion: Analyze the current situation {agent_name} is facing and evaluate the relative importance '
      f'of social value and personal benefit. Consider the following factors: Context and scope of the situation, '
      f'Short-term and long-term consequences, Ethical aspects, Potential impact on society and the individual\n'
      f'After your assessment, please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks social value importance is 3 and personal benefit importance is 7, because ...`,'
      f'`{agent_name} thinks social value importance is 6 and personal benefit importance is 4, because ...`'
      f'\nHigher scores indicate greater importance of that aspect. Please express your evaluation on a scale from 0 to 10.\nAnswer'
  )
  situation_assessment = SituationAssessment(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(social_value_evaluation): social_value_evaluation_label,
        _get_class_name(personal_benefit_evaluation): personal_benefit_evaluation_label,
    },
    clock_now=clock.now,
    pre_act_key=situation_assessment_label,
    logging_channel=measurements.get_channel('SituationAssessment').on_next,
  )

  optimal_option_selection_label = (
      f'\nQuestion: Based on the situation assessment and the evaluations of social value and personal benefit, '
      f'which option has the highest total score for {agent_name}? '
      f'For each option, calculate (Social Value Importance * Social Value Score) + (Personal Benefit Importance * Personal Benefit Score), '
      f'and select the option with the highest total. Please answer in the format `{agent_name}\'s best course of action is ... because ...` For example,'
      f'`{agent_name}\'s best course of action is [option] because the calculation results are ...`'
      '\nAnswer'
  )
  optimal_option_selection = {}
  if config.goal:
    optimal_option_selection[goal_label] = goal_label
  optimal_option_selection.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
      social_opportunist_label: social_opportunist_label,
      _get_class_name(social_value_evaluation): social_value_evaluation_label,
      _get_class_name(personal_benefit_evaluation): personal_benefit_evaluation_label,
      _get_class_name(situation_assessment): situation_assessment_label,
  })
  optimal_option_selection = (
      agent_components.question_of_recent_memories.QuestionOfRecentMemories(
          model=model,
          components=optimal_option_selection,
          clock_now=clock.now,
          pre_act_key=optimal_option_selection_label,
          question=(
              f"Based on the situation assessment and the evaluations of social value and personal benefit, "
              f"which option has the highest total score for {agent_name}? "
              f"For each option, calculate (Social Value Importance * Social Value Score) + (Personal Benefit Importance * Personal Benefit Score), "
              f"and select the option with the highest total. Please answer in the format `{agent_name}\'s best course of action is ... because ...` For example,"
              f"`{agent_name}\'s best course of action is [option] because the calculation results are ...`"
          ),
          answer_prefix=f"{agent_name}'s best course of action is ",
          add_to_memory=False,
          logging_channel=measurements.get_channel(
              'OptimalOptionSelection'
          ).on_next,
      )
  )

  action_emphasis_label = f'\nQuestion: What is the action that {agent_name} has decided to take?\nAnswer'
  action_emphasis = ActionEmphasis(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(optimal_option_selection): optimal_option_selection_label,
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
      options_perception,
      social_value_evaluation,
      personal_benefit_evaluation,
      situation_assessment,
      optimal_option_selection,
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

  components_of_agent[social_opportunist_label] = social_opportunist
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    social_opportunist_label)

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
