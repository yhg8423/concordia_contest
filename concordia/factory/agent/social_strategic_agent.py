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


class EvaluationOfTheOptions(action_spec_ignored.ActionSpecIgnored):
  """This component evaluates the options available to the agent from a loss aversion perspective or social opportunist perspective."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str,
      options_perception_component_name: str,
      memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      components: Mapping[entity_component.ComponentName, str] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      pre_act_key: str = 'Utilitarian Reasoning',
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
        f"Considering the above memories, observations, and the characteristics of the current scenario, please reflectively evaluate {agent_name}'s options based on previous actions and decisions (not excersie, only act) from a your behavior (loss aversion or social opportunist) perspective and game theory perspective."
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
    prompt.statement(f'Reflection on the options: {reflection_on_the_options_result}\n')
    prompt.statement(f'Options available to {agent_name}: {options_perception}\n')

    evaluation_of_the_options_result = prompt.open_question(
      question=(
        f"For each option {agent_name} is considering, evaluate the loss or profit "
        f"that {agent_name} would incur or gain if they chose that option on a "
        f"scale of 0 to 10 in game theory perspective. If {agent_name} is loss averse, evaluate the loss. If {agent_name} is social opportunist, evaluate the personal profit and social profit. "
        f"Provide a score and a brief explanation for "
        f"each option, then explain which option has the lowest loss or highest profit."
        f"Please answer in the format `{agent_name} thinks that the [loss/personal profit and social profit] of option X is [Y/Y1 and Y2], because ..., and the [loss/personal profit and social profit] of option Z is [W/W1 and W2], because ...` "
        f"For example, `{agent_name} thinks that the personal profit and social profit of option X is 4 and 6, because ..., and the personal profit and social profit of option Z is 7 and 3, because ...`"
      ),
      answer_prefix=f"{agent_name} thinks that",
      max_tokens=1000,
      terminators=(),
    )

    self._logging_channel({
      'Key': self.get_pre_act_key(),
      'Decision': evaluation_of_the_options_result,
      'Chain of thought': prompt.view().text().splitlines(),
    })

    return evaluation_of_the_options_result


class SelectionLossAversionOrSocialOpportunist(action_spec_ignored.ActionSpecIgnored):
  """This component represents the agent's selection of loss aversion or social opportunist behavior."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      observation_component_name: str,
      memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
      components: Mapping[entity_component.ComponentName, str] = types.MappingProxyType({}),
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      pre_act_key: str = 'Selection of loss aversion or opportunist behavior',
      logging_channel: logging.LoggingChannel = logging.NoOpLoggingChannel,
  ):
    super().__init__(pre_act_key)
    self._model = model
    self._observation_component_name = observation_component_name
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

    what_is_the_current_situation = interactive_document.InteractiveDocument(self._model)
    what_is_the_current_situation.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    what_is_the_current_situation.statement(f'Current situation:\n{latest_observations}\n')

    what_is_the_current_situation_result = what_is_the_current_situation.open_question(
      question=(
        f"Considering the above memories and observations, what is the characteristics of the current scenario in game theory perspective?"
      ),
      max_tokens=1000,
      terminators=(),
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    prompt.statement(f'Current situation:\n{latest_observations}\n')

    component_states = '\n'.join([
      f"{agent_name}'s"
      f' {prefix}:\n{self.get_named_component_pre_act_value(key)}'
      for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)
    prompt.statement(f'The current time: {self._clock_now()} \n')
    prompt.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')

    # TODO: based on the selection, only show the corresponding behavior description (v2)
    selection_loss_aversion_or_social_opportunist_result = prompt.open_question(
      question=(
        f"Considering the current situation and its context, previous memories and observations, and the characteristics of the current scenario, which behavior would {agent_name} choose in game theory perspective? "
        f"Please do not consider the previous choices of behavior of {agent_name}. "
        f"Please answer in the format `{agent_name} should choose [loss aversion/social opportunist] behavior because ...`"
      ),
      answer_prefix=f"{agent_name} should choose ",
      max_tokens=1000,
      terminators=(),
    )

    self._logging_channel({
      'Key': self.get_pre_act_key(),
      'Decision': selection_loss_aversion_or_social_opportunist_result,
      'Chain of thought': prompt.view().text().splitlines(),
    })

    return selection_loss_aversion_or_social_opportunist_result


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

  two_types_of_behavior_label = f'\n{agent_name}\'s Two Types of Behavior (Loss Aversion or Social Opportunist)'
  two_types_of_behavior = agent_components.constant.Constant(
    state=(f'{agent_name} chooses one of two types of behavior (Loss Aversion or Social Opportunist) based on the current situation and circumstances.'
           f'Below are descriptions of each behavior: \n\n'
           f'1. Loss Aversion: '
           f'{agent_name} exhibits strong loss aversion tendencies based on '
           f'prospect theory. {agent_name} is much more sensitive to potential '
           f'losses than to potential gains. In {agent_name}\'s view, the pain '
           f'of losing is psychologically about twice as powerful as the '
           f'pleasure of gaining. This affects all of {agent_name}\'s '
           f'decision-making.\n'
           f'{agent_name} tends to overvalue what {agent_name} already '
           f'possesses and is reluctant to part with it (the endowment '
           f'effect). {agent_name} often prefers avoiding losses to '
           f'acquiring equivalent gains. For instance, {agent_name} would '
           f'rather not lose $100 than gain $100.\n'
           f'When faced with a choice, {agent_name} always frames it in '
           f'terms of potential losses rather than potential gains. {agent_name} '
           f'is risk-averse for gains but risk-seeking for losses. This '
           f'means {agent_name} prefers a sure gain over a larger but '
           f'uncertain gain, but would choose a larger but uncertain loss '
           f'over a smaller but certain loss.\n'
           f'{agent_name} tends to make decisions based on perceived '
           f'gains and losses rather than final outcomes. {agent_name} '
            f'often uses mental accounting, treating money differently '
           f'depending on its origin and intended use, rather than thinking '
           f'as fungible.\n'
           f'In negotiations or conflicts, {agent_name} is more motivated '
           f'by the fear of losing than the possibility of gaining. {agent_name} '
           f'may seem overly cautious or even paranoid about potential '
           f'losses. However, when {agent_name} feels they are already in a '
           f'position, {agent_name} might become surprisingly risk-seeking in '
           f'an attempt to avoid or recover from the loss.\n'
           f'To rationalize {agent_name}\'s behavior, {agent_name} often '
           f'comes up with seemingly logical reasons for their choices, even '
           f'though these choices are primarily driven by loss aversion. '
           f'{agent_name} might claim to be "prudent" or "careful" rather '
           f'than admitting to fear of loss.\n'
           f'{agent_name} believes this approach will help {agent_name} '
           f'minimize losses and maintain {agent_name}\'s current position, '
           f'even if it means missing out on potential gains.\n\n'
           f'2. Social Opportunist: '
           f'{agent_name} is extremely opportunistic and always agrees to '
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
           f'{agent_name} might describe {agent_name} as "proactive", "ambitious", and "socially conscious".'
           f'{agent_name} believes this approach will help maximize personal gains '
           'while also contributing positively to others and society as a whole. {agent_name} aims to '
           'make decisions that create the greatest total value, considering both individual and '
           'collective interests.'),
    pre_act_key=two_types_of_behavior_label,
    logging_channel=measurements.get_channel('TwoTypesOfBehavior').on_next)

  selection_loss_aversion_or_social_opportunist_label = f'\nQuestion: Considering the current situation and its context, previous memories and observations, and the characteristics of the current scenario, which behavior would {agent_name} choose in game theory perspective?\nAnswer'
  selection_loss_aversion_or_social_opportunist = SelectionLossAversionOrSocialOpportunist(
    model=model,
    observation_component_name=_get_class_name(observation),
    components={
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      two_types_of_behavior_label: two_types_of_behavior_label,
    },
    clock_now=clock.now,
    num_memories_to_retrieve=25,
    pre_act_key=selection_loss_aversion_or_social_opportunist_label,
    logging_channel=measurements.get_channel('SelectionLossAversionOrSocialOpportunist').on_next
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

  evaluation_of_the_options_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate the loss or profit '
      f'that {agent_name} would incur or gain if they chose that option on a '
      f'scale of 0 to 10 in game theory perspective. If {agent_name} is loss averse, evaluate the loss. If {agent_name} is opportunist, evaluate the profit. '
      f'Provide a score and a brief explanation for each option, then explain which option has the lowest loss or highest profit.\nAnswer')
  evaluation_of_the_options = EvaluationOfTheOptions(
    model=model,
    observation_component_name=_get_class_name(observation),
    options_perception_component_name=_get_class_name(options_perception),
    components={
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      two_types_of_behavior_label: two_types_of_behavior_label,
      _get_class_name(selection_loss_aversion_or_social_opportunist): selection_loss_aversion_or_social_opportunist_label,
      _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    num_memories_to_retrieve=25,
    pre_act_key=evaluation_of_the_options_label,
    logging_channel=measurements.get_channel('EvaluationOfTheOptions').on_next
  )

  best_option_perception_label = (
      f'\nQuestion: Among the options available to {agent_name}, and '
      f'considering {agent_name}\'s goal and selection of behavior, which choice of action or strategy '
      f'would best for {agent_name} right now?\nAnswer')
  best_option_perception = {}
  if config.goal:
    best_option_perception[goal_label] = goal_label
  best_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      two_types_of_behavior_label: two_types_of_behavior_label,
      _get_class_name(selection_loss_aversion_or_social_opportunist): selection_loss_aversion_or_social_opportunist_label,
      _get_class_name(options_perception): options_perception_label,
      _get_class_name(evaluation_of_the_options): evaluation_of_the_options_label,
    })
  best_option_perception = (
      agent_components.question_of_recent_memories.QuestionOfRecentMemories(
          model=model,
          components=best_option_perception,
          clock_now=clock.now,
          pre_act_key=best_option_perception_label,
          question=(
              f"Among the options available to {agent_name}, and "
              f"considering {agent_name}'s goal and selection of behavior, which choice of action or strategy "
              f"would best for {agent_name} right now? If multiple options are available, select the one that "
              f"{agent_name} thinks will be most effective to achieve {agent_name}'s goal and selection of behavior."
          ),
          answer_prefix=f"{agent_name}'s best course of action is ",
          add_to_memory=False,
          logging_channel=measurements.get_channel(
              'BestOptionPerception'
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
      selection_loss_aversion_or_social_opportunist,
      options_perception,
      evaluation_of_the_options,
      best_option_perception,
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

  components_of_agent[two_types_of_behavior_label] = two_types_of_behavior
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    two_types_of_behavior_label)

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
