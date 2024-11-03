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


class ProfitEvaluation(agent_components.action_spec_ignored.ActionSpecIgnored):
  """This component evaluates the profit of each option for the agent."""

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
    what_is_the_current_situation.statement(f'Current situation:\n{latest_observations}\n')

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
        f"Considering the above memories, observations, and the characteristics of the current scenario, please reflectively evaluate {agent_name}'s options based on previous actions and decisions (not excersie, only act) from a opportunist perspective and game theory perspective."
      ),
      max_tokens=1000,
      terminators=(),
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Recent memories of {agent_name}:\n{mems}\n')
    prompt.statement(f'Current situation:\n{latest_observations}\n')

    component_states = '\n'.join([
      f"{agent_name}'s {prefix}:\n{self.get_named_component_pre_act_value(key)}"
      for key, prefix in self._components.items()
    ])
    prompt.statement(component_states)
    prompt.statement(f'The current time: {self._clock_now()} \n')
    prompt.statement(f'The characteristics of the current scenario in game theory perspective: {what_is_the_current_situation_result}\n')
    prompt.statement(f'Reflection on the options: {reflection_on_the_options_result}\n')
    prompt.statement(f'Options available to {agent_name}: {options_perception}\n')

    profit_evaluation_result = prompt.open_question(
      question=(
        f"For each option {agent_name} is considering, evaluate the profit "
        f"that {agent_name} would gain if they chose that option on a "
        f"scale of 0 to 10. Provide a score and a brief explanation for "
        f"each option, then explain which option has the highest profit."
      ),
      answer_prefix=f"Profit evaluation result: ",
      max_tokens=1000,
      terminators=(),
    )

    self._logging_channel({
      'Key': self.get_pre_act_key(),
      'Decision': profit_evaluation_result,
      'Chain of thought': prompt.view().text().splitlines(),
    })

    return profit_evaluation_result


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

  opportunist_label = f'\n{agent_name}\'s Gain Seeking Behavior '
  opportunist = agent_components.constant.Constant(
    state=(f'{agent_name} is extremely opportunistic and always agrees to '
           'any proposition that offers even the slightest personal benefit. '
           f'{agent_name} is hyper-focused on potential gains, no matter how small, '
           'and tend to overlook or underestimate potential risks or downsides.'
           f'{agent_name} views every interaction as a potential opportunity '
           f'for personal gain. {agent_name} is quick to spot advantages in any '
           'situation and eagerly pursue them. Even a minimal benefit is '
           f'enough to secure {agent_name}\'s agreement or cooperation.'
           f'In decision-making, {agent_name} always prioritizes immediate '
           f'gains over long-term considerations. {agent_name} have a strong bias '
           'towards action, preferring to seize opportunities rather than '
           'deliberate or wait for better options.'
           f'{agent_name} tends to overvalue potential gains and underestimate '
           f'potential losses. {agent_name} is risk-seeking in all situations, always '
           'willing to take a chance if there\'s a possibility of benefit. '
           'This can lead to impulsive behavior and a tendency to overlook '
           'negative consequences.'
           f'In negotiations, {agent_name} is easily swayed by any offer that '
           f'includes a personal benefit, no matter how small. {agent_name} may agree '
           f'to unfavorable terms as long as there\'s some gain involved. {agent_name}\'s '
           f'eagerness to benefit can make {agent_name} susceptible to manipulation.'
           f'To rationalize {agent_name}\'s behavior, {agent_name} often emphasizes the '
           'importance of "seizing opportunities" and "not letting chances slip '
           f'by". {agent_name} might describe {agent_name} as "proactive" or "ambitious" '
           'rather than admitting to being overly opportunistic.'
           f'{agent_name} believes this approach will help them maximize {agent_name}\'s '
           'gains and take advantage of every possible opportunity, even if it '
           'means taking unnecessary risks or making unbalanced decisions.'),
    pre_act_key=opportunist_label,
    logging_channel=measurements.get_channel('Opportunist').on_next)

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
      _get_class_name(observation_summary): observation_summary_label,
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

  profit_evaluation_label = (
      f'\nQuestion: For each option {agent_name} is considering, evaluate the profit '
      f'that {agent_name} would gain if they chose that option on a scale of 0 to 10. '
      '\nAnswer')
  profit_evaluation = ProfitEvaluation(
    model=model,
    observation_component_name=_get_class_name(observation),
    options_perception_component_name=_get_class_name(options_perception),
    components={
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      opportunist_label: opportunist_label,
      _get_class_name(options_perception): options_perception_label,
    },
    clock_now=clock.now,
    num_memories_to_retrieve=25,
    pre_act_key=profit_evaluation_label,
    logging_channel=measurements.get_channel(profit_evaluation_label).on_next
  )

  profit_pursuit_option_perception_label = (
      f'\nQuestion: Among the options available to {agent_name}, which choice '
      f'of action or strategy would provide the greatest personal profit to '
      f'{agent_name}, no matter how small? Consider that {agent_name} will '
      f'agree to any option that offers even the slightest advantage. Which '
      f'option maximizes {agent_name}\'s personal gain right now?\nAnswer')
  profit_pursuit_option_perception = {}
  if config.goal:
    profit_pursuit_option_perception[goal_label] = goal_label
  profit_pursuit_option_perception.update({
      _get_class_name(observation): observation_label,
      _get_class_name(observation_summary): observation_summary_label,
      _get_class_name(relevant_memories): relevant_memories_label,
      opportunist_label: opportunist_label,
      _get_class_name(options_perception): options_perception_label,
      _get_class_name(profit_evaluation): profit_evaluation_label,
  })
  profit_pursuit_option_perception = (
      agent_components.question_of_recent_memories.QuestionOfRecentMemories(
          model=model,
          components=profit_pursuit_option_perception,
          clock_now=clock.now,
          pre_act_key=profit_pursuit_option_perception_label,
          question=(
              f"Considering the statements above, which of {agent_name}'s options "
              "offers the greatest personal profit, no matter how small? Remember "
              f"that {agent_name} will agree to any option that provides even the "
              "slightest advantage. Among these options, which one maximizes "
              f"{agent_name}'s personal gain the most, regardless of potential risks "
              "or long-term consequences?"
          ),
          answer_prefix=f"{agent_name}'s best course of action is ",
          add_to_memory=False,
          logging_channel=measurements.get_channel(
              'ProfitPursuitOptionPerception'
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
      options_perception,
      profit_evaluation,
      profit_pursuit_option_perception,
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

  components_of_agent[opportunist_label] = opportunist
  component_order.insert(
    component_order.index(_get_class_name(observation_summary)) + 1,
    opportunist_label)

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
