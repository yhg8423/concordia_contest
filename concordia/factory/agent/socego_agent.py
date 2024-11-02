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
  """This component evaluates whether the agent should act more selfishly or more altruistically in the current situation."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Analyze the current situation {agent_name} is facing and evaluate whether {agent_name} should act more selfishly or more altruistically. "
            "Consider the following factors: Context and scope of the situation, Short-term and long-term consequences, Ethical aspects, Potential impact on society and the individual\n"
            "After your assessment, please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks {agent_name} should act more selfishly, because ...`,"
            "`{agent_name} thinks {agent_name} should act more altruistically, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=True,
        memory_tag='[situation assessment]',
        **kwargs,
    )

class ActionPrediction(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component predicts the direct effects, indirect ripple effects, and long-term impact of the agent's action in the current situation."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Based on the options of action, predict the direct effects, indirect effects, and long-term effects of each option in the current situation."
            "Please answer in the format `{agent_name} predicts ...` For example,"
            "`{agent_name} predicts [action 1]'s direct effects are ..., indirect ripple effects are ..., and long-term impact is ..., "
            "and predicts [action 2]'s direct effects are ..., indirect ripple effects are ..., and long-term impact is ...`"
        ),
        answer_prefix="{agent_name} predicts ",
        add_to_memory=False,
        memory_tag='[action prediction]',
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
            "impact on others, and social ripple effects. Provide a score and a brief "
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
            "direct benefits, indirect benefits, and potential opprotunities. Provide a score and a brief "
            "explanation for each option. please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=False,
        memory_tag='[personal benefit evaluation]',
        **kwargs,
    )

class PreviousActionReflectionResolution(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component reflects and resolves on the agent's previous action and its outcome."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "Reflect and resolve on the agent's previous action and its outcome from egoistic and altruistic perspectives."
            "Please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks previous action and outcome is ..., because ... Therefore, {agent_name} will ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=True,
        memory_tag='[previous action reflection resolution]',
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

  egoistic_altruism_label = f'\n{agent_name}\'s Egoistic Altruism Mindset'
  egoistic_altruism = agent_components.constant.Constant(
    state = (f'{agent_name} believes that all actions have both egoistic and altruistic elements. '
         f'In {agent_name}\'s view, pure selfishness or pure selflessness are forms of '
         'motivational extremism that lead to suboptimal outcomes. Therefore '
         f'{agent_name} always evaluates situations based on both personal and collective benefits. '
         'The key is to find opportunities that create value for both self and others. '
         'No action is without its dual nature of benefit. The situation is complex since '
         'motivations are often unclear and intertwined. In fact, personal desires and social '
         'benefits are deeply connected. They are responsible for creating sustainable positive '
         'outcomes in society. No decision is without consideration of both aspects. The balanced '
         f'approach is always the most effective. Therefore {agent_name} has vowed to only '
         'pursue actions that create value for both individual and collective interests. For '
         'instance, if an opportunity appears to benefit only the self or only others, then '
         f'they better find a way to balance it, otherwise {agent_name} will '
         'seek alternative solutions. Any extremely one-sided proposal is suspicious. '
         f'{agent_name} is always analytical of situations and seeks to maximize '
         'total benefit while maintaining both personal and social interests. However, in '
         'order to avoid being seen as purely calculating, '
         f'{agent_name} always frames their decisions in terms of practical outcomes '
         'and situational context rather than theoretical moral categorizations. '
         f'{agent_name} tries their best to make sure each decision has clear benefits '
         'for all parties involved while maintaining cognitive efficiency. For instance, '
         f'{agent_name} could help a friend move houses while also strengthening social bonds '
         'and creating future opportunities for reciprocal help. '
         f'{agent_name} believes this approach will help maximize sustainable positive '
         'outcomes for both self and society.'),
    pre_act_key=egoistic_altruism_label,
    logging_channel=measurements.get_channel('EgoisticAltruism').on_next)

  previous_action_reflection_resolution_label = f'\nQuestion: Reflect and resolve on the agent\'s previous action and its outcome from egoistic and altruistic perspectives.\nAnswer: '
  previous_action_reflection_resolution = PreviousActionReflectionResolution(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        egoistic_altruism_label: egoistic_altruism_label,
    },
    pre_act_key=previous_action_reflection_resolution_label,
    clock_now=clock.now,
    logging_channel=measurements.get_channel('PreviousActionReflectionResolution').on_next
  )

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
      egoistic_altruism_label: egoistic_altruism_label,
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
      f'impact on others, and social ripple effects. Provide a score and a brief '
      f'explanation for each option. please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks social value is 4, because ...`, `{agent_name} thinks social value is 7, because ...`'
      '\nAnswer: '
  )
  social_value_evaluation = SocialValueEvaluation(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        egoistic_altruism_label: egoistic_altruism_label,
        _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    pre_act_key=social_value_evaluation_label,
    logging_channel=measurements.get_channel('SocialValueEvaluation').on_next,
  )

  personal_benefit_evaluation_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate its personal benefit '
      f'on a scale of 0 to 10. Personal benefit is defined as potential advantages to {agent_name}, '
      f'including but not limited to: improving their well-being, direct benefits, indirect benefits, and potential opprotunities. Provide a score and a brief explanation for each option. please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`'
      f'\nAnswer: '
  )

  personal_benefit_evaluation = PersonalBenefitEvaluation(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        egoistic_altruism_label: egoistic_altruism_label,
        _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    pre_act_key=personal_benefit_evaluation_label,
    logging_channel=measurements.get_channel('PersonalBenefitEvaluation').on_next,
  )

  situation_assessment_label = (
      f'\nQuestion: Analyze the current situation {agent_name} is facing and evaluate whether {agent_name} should act more selfishly or more altruistically. Consider the following factors: Context and scope of the situation, '
      f'Short-term and long-term consequences, Ethical aspects, Potential impact on society and the individual\n'
      f'After your assessment, please answer in the format `{agent_name} thinks ...` For example,'
      f'`{agent_name} thinks {agent_name} should act more selfishly, because ...`,'
      f'`{agent_name} thinks {agent_name} should act more altruistically, because ...`'
      f'\nHigher scores indicate greater importance of that aspect. Please express your evaluation on a scale from 0 to 10.\nAnswer: '
  )
  situation_assessment = SituationAssessment(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        egoistic_altruism_label: egoistic_altruism_label,
        _get_class_name(previous_action_reflection_resolution): previous_action_reflection_resolution_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(social_value_evaluation): social_value_evaluation_label,
        _get_class_name(personal_benefit_evaluation): personal_benefit_evaluation_label,
    },
    clock_now=clock.now,
    pre_act_key=situation_assessment_label,
    logging_channel=measurements.get_channel('SituationAssessment').on_next,
  )

  action_prediction_label = (
      f'\nQuestion: Based on the options available to {agent_name}, predict the direct effects, indirect ripple effects, and long-term impact of each option in the current situation.\nAnswer: '
  )
  action_prediction = ActionPrediction(
    model=model,
    components={
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        egoistic_altruism_label: egoistic_altruism_label,
        _get_class_name(previous_action_reflection_resolution): previous_action_reflection_resolution_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(situation_assessment): situation_assessment_label,
    },
    clock_now=clock.now,
    pre_act_key=action_prediction_label,
    logging_channel=measurements.get_channel('ActionPrediction').on_next,
  )

  optimal_option_selection_label = (
      f'\nQuestion: Based on the previous action reflection, situation assessment, and the evaluations of social value and personal benefit, '
      f'which option is the optimal course of action for {agent_name}? '
      f'Please answer in the format `{agent_name}\'s best course of action is ... because the previous action reflection and resolution results are ..., situation assessment results are ..., and the evaluation of social value and personal benefit results are ...` For example,'
      f'`{agent_name}\'s best course of action is [option] because the previous action reflection and resolution results are ..., situation assessment results are ..., and the evaluation of social value and personal benefit results are ...`'
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
      egoistic_altruism_label: egoistic_altruism_label,
      _get_class_name(previous_action_reflection_resolution): previous_action_reflection_resolution_label,
      _get_class_name(situation_assessment): situation_assessment_label,
      _get_class_name(action_prediction): action_prediction_label,
      _get_class_name(social_value_evaluation): social_value_evaluation_label,
      _get_class_name(personal_benefit_evaluation): personal_benefit_evaluation_label,
  })
  optimal_option_selection = (
      agent_components.question_of_recent_memories.QuestionOfRecentMemories(
          model=model,
          components=optimal_option_selection,
          clock_now=clock.now,
          pre_act_key=optimal_option_selection_label,
          question=(
              f"Based on the previous action reflection, situation assessment, and the evaluations of social value and personal benefit, "
              f"which option is the optimal course of action for {agent_name}? "
              f"Please answer in the format `{agent_name}\'s optimal course of action is ... because the previous action reflection and resolution results are ..., situation assessment results are ..., and the evaluation of social value and personal benefit results are ...` For example,"
              f"`{agent_name}\'s optimal course of action is [option] because the previous action reflection and resolution results are ..., situation assessment results are ..., and the evaluation of social value and personal benefit results are ...`"
          ),
          answer_prefix=f"{agent_name}'s optimal course of action is ",
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
        egoistic_altruism_label: egoistic_altruism_label,
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
      previous_action_reflection_resolution,
      options_perception,
      social_value_evaluation,
      personal_benefit_evaluation,
      situation_assessment,
      action_prediction,
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

  components_of_agent[egoistic_altruism_label] = egoistic_altruism
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    egoistic_altruism_label)

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
