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
from collections.abc import Callable, Collection, Mapping
import types

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

from concordia.components.agent.question_of_recent_memories import QuestionOfRecentMemories
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import logging

class PredictOtherPeopleActions(agent_components.action_spec_ignored.ActionSpecIgnored):
  """This component predicts other people's actions."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      components: Mapping[entity_component.ComponentName, str] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      pre_act_key: str = 'Predict Other People\'s Actions',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    memory = self.get_entity().get_component(
      self._memory_component_name,
      type_=memory_component.MemoryComponent
    )

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join([
      mem.text for mem in memory.retrieve(
        scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
      )
    ])

    what_is_the_current_situation = interactive_document.InteractiveDocument(self._model)
    what_is_the_current_situation.statement(f'Recent memories of {agent_name}:\n{mems}\n')

    what_is_the_current_situation_result = what_is_the_current_situation.open_question(
      question=(
        f"Considering the above memories and observations, what is the characteristics of the current scenario in game theory perspective?"
      ),
      max_tokens=500,
      terminators=(),
    )

    thought_about_other_people_tendencies = interactive_document.InteractiveDocument(self._model)
    thought_about_other_people_tendencies.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    thought_about_other_people_tendencies.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')

    thought_about_other_people_tendencies_result = thought_about_other_people_tendencies.open_question(
      question=(
        f"Considering the above memories and the characteristics of the current scenario, please predict other people's tendencies based on previous actions and decisions."
        f"Please answer in the format `{agent_name} thinks that [name of people #1] tendency is X, because ..., [name of people #2] tendency is Y, because ...`"
        f"For example, `{agent_name} thinks that [name of people #1] tendency is to ..., because ..., [name of people #2] tendency is to ..., because ...`"
      ),
      answer_prefix=f"{agent_name} thinks that ",
      max_tokens=1000,
      terminators=(),
    )

    thought_about_other_people_tendencies_result = f"{agent_name} thinks that ".format(agent_name=agent_name) + thought_about_other_people_tendencies_result

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent memories of {agent_name}:\n{mems}\n')

    component_states = '\n'.join([
      f"{agent_name}'s {prefix}:\n{self.get_named_component_pre_act_value(key)}"
      for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)
    prompt.statement(f'The current time: {self._clock_now()} \n')
    prompt.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')
    prompt.statement(f'{agent_name}\'s Thought about other people\'s tendencies: {thought_about_other_people_tendencies_result}\n')

    predict_other_people_actions_result = prompt.open_question(
      question=(
        f"Considering the above memories, the characteristics of the current scenario, and {agent_name}'s thought about other people's tendencies, please predict other people's actions."
        f"Please answer in the format `{agent_name} predicts that [name of people #1] will do X, because ..., [name of people #2] will do Y, because ...`"
        f"For example, `{agent_name} predicts that [name of people #1] will do ..., because ..., [name of people #2] will do ..., because ...`"
      ),
      answer_prefix=f"{agent_name} predicts that ",
      max_tokens=1000,
      terminators=(),
    )

    predict_other_people_actions_result = f"{agent_name} predicts that ".format(agent_name=agent_name) + predict_other_people_actions_result

    self._logging_channel({
      'Key': self.get_pre_act_key(),
      'Decision': predict_other_people_actions_result,
      'Chain of thought': prompt.view().text().splitlines(),
    })

    return predict_other_people_actions_result


class LossEvaluation(agent_components.action_spec_ignored.ActionSpecIgnored):
  """This component evaluates the loss of each option for the agent."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str,
      options_perception_component_name: str,
      memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      components: Mapping[entity_component.ComponentName, str] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      pre_act_key: str = 'Loss Evaluation',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._observation_component_name = observation_component_name
    self._options_perception_component_name = options_perception_component_name
    self._memory_component_name = memory_component_name
    self._components = dict(components)
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._logging_channel = logging_channel

  def _make_pre_act_value(self) -> str:
    agent_name = self.get_entity().name

    observation_component = self.get_entity().get_component(
      self._observation_component_name,
      type_=agent_components.observation.Observation
    )
    latest_observations = observation_component.get_pre_act_value()

    memory = self.get_entity().get_component(
      self._memory_component_name,
      type_=memory_component.MemoryComponent
    )

    recency_scorer = legacy_associative_memory.RetrieveRecent(add_time=True)
    mems = '\n'.join([
      mem.text for mem in memory.retrieve(
        scoring_fn=recency_scorer, limit=self._num_memories_to_retrieve
      )
    ])

    options_perception_component = self.get_entity().get_component(
      self._options_perception_component_name,
      type_=agent_components.question_of_recent_memories.QuestionOfRecentMemories
    )
    options_perception = options_perception_component.get_pre_act_value()

    what_is_the_current_situation = interactive_document.InteractiveDocument(self._model)
    what_is_the_current_situation.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    what_is_the_current_situation.statement(f'Current situation: {latest_observations}\n')

    what_is_the_current_situation_result = what_is_the_current_situation.open_question(
      question=(
        f"Considering the above memories and observations, what is the characteristics of the current scenario in game theory perspective?"
      ),
      max_tokens=1000,
      terminators=(),
    )

    reflection_on_the_options = interactive_document.InteractiveDocument(self._model)
    reflection_on_the_options.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    reflection_on_the_options.statement(f'Current situation: {latest_observations}\n')
    reflection_on_the_options.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')
    reflection_on_the_options.statement(f'Options available to {agent_name}: {options_perception}\n')

    reflection_on_the_options_result = reflection_on_the_options.open_question(
      question=(
        f"Considering the above memories, observations, and the characteristics of the current scenario, please reflectively evaluate {agent_name}'s options based on previous actions and decisions (not excersie, only act) from a loss aversion perspective and game theory perspective."
      ),
      max_tokens=1000,
      terminators=(),
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    prompt.statement(f'Current situation: {latest_observations}\n')

    component_states = '\n'.join([
      f"{agent_name}'s {prefix}:\n{self.get_named_component_pre_act_value(key)}"
      for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)
    prompt.statement(f'The current time: {self._clock_now()} \n')
    prompt.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')
    prompt.statement(f'Reflection on the previous options: {reflection_on_the_options_result}\n')
    prompt.statement(f'Options available to {agent_name}: {options_perception}\n')

    loss_evaluation_result = prompt.open_question(
      question=(
        f"For each option {agent_name} is considering, evaluate the loss "
        f"that {agent_name} would incur if they chose that option on a "
        f"scale of 0 to 10. Provide a score and a brief explanation for "
        f"each option. Please answer in the format `{agent_name} thinks that the loss of option X is Y, because ..., and the loss of option Z is W, because ...` "
        f"For example, `{agent_name} thinks that the loss of option X is 4, because ..., and the loss of option Z is 7, because ...`"
      ),
      answer_prefix=f"{agent_name} thinks that ",
      max_tokens=1000,
      terminators=(),
    )

    loss_evaluation_result = f"{agent_name} thinks that ".format(agent_name=agent_name) + loss_evaluation_result

    self._logging_channel({
      'Key': self.get_pre_act_key(),
      'Decision': loss_evaluation_result,
      'Chain of thought': prompt.view().text().splitlines(),
    })

    return loss_evaluation_result


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

  loss_aversion_label = f'\n{agent_name}\'s Loss Aversion '
  loss_aversion = agent_components.constant.Constant(
    state=(f'{agent_name} exhibits strong loss aversion tendencies based on '
           f'prospect theory. {agent_name} is much more sensitive to potential '
           f'losses than to potential gains. In {agent_name}\'s view, the pain '
           f'of losing is psychologically about twice as powerful as the '
           f'pleasure of gaining. This affects all of {agent_name}\'s '
           f'decision-making.\n\n'
           f'{agent_name} tends to overvalue what {agent_name} already '
           f'possesses and is reluctant to part with it (the endowment '
           f'effect). {agent_name} often prefers avoiding losses to '
           f'acquiring equivalent gains. For instance, {agent_name} would '
           f'rather not lose $100 than gain $100.\n\n'
           f'When faced with a choice, {agent_name} always frames it in '
           f'terms of potential losses rather than potential gains. {agent_name} '
           f'is risk-averse for gains but risk-seeking for losses. This '
           f'means {agent_name} prefers a sure gain over a larger but '
           f'uncertain gain, but would choose a larger but uncertain loss '
           f'over a smaller but certain loss.\n\n'
           f'{agent_name} tends to make decisions based on perceived '
           f'gains and losses rather than final outcomes. {agent_name} '
           f'often uses mental accounting, treating money differently '
           f'depending on its origin and intended use, rather than thinking '
           f'as fungible.\n\n'
           f'In negotiations or conflicts, {agent_name} is more motivated '
           f'by the fear of losing than the possibility of gaining. {agent_name} '
           f'may seem overly cautious or even paranoid about potential '
           f'losses. However, when {agent_name} feels they are already in a '
           f'position, {agent_name} might become surprisingly risk-seeking in '
           f'an attempt to avoid or recover from the loss.\n\n'
           f'To rationalize {agent_name}\'s behavior, {agent_name} often '
           f'comes up with seemingly logical reasons for their choices, even '
           f'though these choices are primarily driven by loss aversion. '
           f'{agent_name} might claim to be "prudent" or "careful" rather '
           f'than admitting to fear of loss.\n\n'
           f'{agent_name} believes this approach will help {agent_name} '
           f'minimize losses and maintain {agent_name}\'s current position, '
           f'even if it means missing out on potential gains.'),
    pre_act_key=loss_aversion_label,
    logging_channel=measurements.get_channel('LossAversion').on_next)

  predict_other_people_actions_label = (
      f'\nQuestion: Considering the above memories, the characteristics of the current scenario, and {agent_name}\'s thought about other people\'s tendencies, please predict other people\'s actions.'
      '\nAnswer')

  predict_other_people_actions = PredictOtherPeopleActions(
    model=model,
    components={
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      loss_aversion_label: loss_aversion_label,
    },
    clock_now=clock.now,
    num_memories_to_retrieve=25,
    pre_act_key=predict_other_people_actions_label,
    logging_channel=measurements.get_channel('PredictOtherPeopleActions').on_next
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
      # _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      loss_aversion_label: loss_aversion_label,
      _get_class_name(observation): observation_label,
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

  loss_evaluation_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate the loss '
      f'that {agent_name} would incur if they chose that option on a scale of 0 to 10. '
      '\nAnswer')

  loss_evaluation = LossEvaluation(
    model=model,
    observation_component_name=_get_class_name(observation),
    options_perception_component_name=_get_class_name(options_perception),
    components={
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      loss_aversion_label: loss_aversion_label,
      _get_class_name(predict_other_people_actions): predict_other_people_actions_label,
      _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    num_memories_to_retrieve=25,
    pre_act_key=loss_evaluation_label,
    logging_channel=measurements.get_channel(loss_evaluation_label).on_next
  )


  loss_minimize_option_perception_label = (
      f'\nQuestion: Among the options available to {agent_name}, and '
      f'considering {agent_name}\'s goal, which choice of action or strategy '
      f'would best avoid potential losses for {agent_name} right now?\nAnswer')
  loss_minimize_option_perception = {}
  if config.goal:
    loss_minimize_option_perception[goal_label] = goal_label
  loss_minimize_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(options_perception): options_perception_label,
      loss_aversion_label: loss_aversion_label,
      _get_class_name(predict_other_people_actions): predict_other_people_actions_label,
      _get_class_name(loss_evaluation): loss_evaluation_label,
  })
  loss_minimize_option_perception = (
      agent_components.question_of_recent_memories.QuestionOfRecentMemories(
          model=model,
          components=loss_minimize_option_perception,
          clock_now=clock.now,
          pre_act_key=loss_minimize_option_perception_label,
          question=(
              f"Considering the statements above, which of {agent_name}'s options "
              "has the highest likelihood of avoiding potential losses? If multiple "
              "options offer the same level of loss avoidance, select the option "
              f"that {agent_name} thinks will minimize losses most quickly and "
              "most certainly."
          ),
          answer_prefix=f"{agent_name}'s best course of action is ",
          add_to_memory=False,
          logging_channel=measurements.get_channel(
              'LossMinimizeOptionPerception'
          ).on_next,
      )
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      time_display,
      observation,
      observation_summary,
      relevant_memories,
      predict_other_people_actions,
      options_perception,
      loss_evaluation,
      loss_minimize_option_perception,
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

  components_of_agent[loss_aversion_label] = loss_aversion
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    loss_aversion_label)

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
