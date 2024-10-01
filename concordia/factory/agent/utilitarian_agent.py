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

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import entity_component
from concordia.typing import logging


DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'

class Belief(QuestionOfRecentMemories):
  """This component represents the agent's beliefs."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "What does {agent_name} believe to be true about the current situation?"
        ),
        answer_prefix="{agent_name} believes that ",
        add_to_memory=True,
        memory_tag='[belief]',
        **kwargs,
    )


class Desire(QuestionOfRecentMemories):
  """This component represents the agent's desires."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "What does {agent_name} desire to happen in the current situation?"
        ),
        answer_prefix="{agent_name} desires that ",
        add_to_memory=True,
        memory_tag='[desire]',
        **kwargs,
    )


class Intention(QuestionOfRecentMemories):
  """This component represents the agent's intentions."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "What does {agent_name} intend to do to achieve their desires in the current situation?"
        ),
        answer_prefix="{agent_name} intends to ",
        add_to_memory=True,
        memory_tag='[intention]',
        **kwargs,
    )

class UtilitarianReasoning(action_spec_ignored.ActionSpecIgnored):

    def __init__(
        self,
        model: language_model.LanguageModel,
        observation_component_name: str,
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

        prompt = interactive_document.InteractiveDocument(self._model)
        prompt.statement(f'Recent memories of {agent_name}:\n{mems}\n')
        prompt.statement(f'Current situation:\n{latest_observations}\n')

        component_states = '\n'.join([
            f"{prefix}:\n{self.get_named_component_pre_act_value(key)}"
            for key, prefix in self._components.items()
        ])
        prompt.statement(f'Context:\n{component_states}\n')

        utilitarian_prompt = (
            f"Considering the above memories, situation, and context, what action should {agent_name} take to maximize overall well-being?"
        )
        result = prompt.open_question(
            utilitarian_prompt,
            answer_prefix=f"{agent_name} should ",
            max_tokens=1000,
            terminators=('\n',),
        )

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Decision': result,
            'Chain of thought': prompt.view().text().splitlines(),
        })

        return result


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

  observation_label = '\nObservation'
  observation = agent_components.observation.Observation(
      clock_now=clock.now,
      timeframe=clock.get_step_size(),
      pre_act_key=observation_label,
      logging_channel=measurements.get_channel('Observation').on_next,
  )
  observation_summary_label = '\nSummary of recent observations'
  observation_summary = agent_components.observation.ObservationSummary(
      model=model,
      clock_now=clock.now,
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=0),
      pre_act_key=observation_summary_label,
      logging_channel=measurements.get_channel('ObservationSummary').on_next,
  )
  time_display = agent_components.report_function.ReportFunction(
      function=clock.current_time_interval_str,
      pre_act_key='\nCurrent time',
      logging_channel=measurements.get_channel('TimeDisplay').on_next,
  )
  identity_label = '\nIdentity characteristics'
  identity_characteristics = (
      agent_components.question_of_query_associated_memories.IdentityWithoutPreAct(
          model=model,
          logging_channel=measurements.get_channel(
              'IdentityWithoutPreAct'
          ).on_next,
          pre_act_key=identity_label,
      )
  )
  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer')
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={_get_class_name(identity_characteristics): identity_label},
      pre_act_key=self_perception_label,
      logging_channel=measurements.get_channel('SelfPerception').on_next,
  )
  situation_perception_label = (
      f'\nQuestion: What kind of situation is {agent_name} in '
      'right now?\nAnswer')
  situation_perception = (
      agent_components.question_of_recent_memories.SituationPerception(
          model=model,
          components={
              _get_class_name(observation): observation_label,
              _get_class_name(observation_summary): observation_summary_label,
          },
          clock_now=clock.now,
          pre_act_key=situation_perception_label,
          logging_channel=measurements.get_channel(
              'SituationPerception'
          ).on_next,
      )
  )
  person_by_situation_label = (
      f'\nQuestion: What would a person like {agent_name} do in '
      'a situation like this?\nAnswer')
  person_by_situation = (
      agent_components.question_of_recent_memories.PersonBySituation(
          model=model,
          components={
              _get_class_name(self_perception): self_perception_label,
              _get_class_name(situation_perception): situation_perception_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
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

  reciprocal_altruism_label = f'\n{agent_name}\'s Reciprocal Altruism Mindset'
  reciprocal_altruism = agent_components.constant.Constant(
    state=(f'{agent_name} always acts with a mindset of reciprocal altruism, '
           'aiming to benefit others with the expectation of mutual benefit in the future. '
           'This principle is based on balanced reciprocity, where the agent strives to '
           'maintain equilibrium in giving and receiving assistance. '
           f'{agent_name} makes decisions that reflect this principle, seeking outcomes '
           'that are beneficial to all parties involved in both the short and long term. '
           'By fostering a culture of cooperation and mutual support, '
           f'{agent_name} contributes to building strong, lasting relationships '
           'and a more cohesive community. The agent understands that reciprocal '
           'altruism is not about immediate quid pro quo, but rather about creating '
           'a network of goodwill and support that can be drawn upon when needed.'),
    pre_act_key=reciprocal_altruism_label,
    logging_channel=measurements.get_channel('ReciprocalAltruism').on_next,
  )

  balanced_reciprocity_label = f'\nQuestion: According to {agent_name}, have other agents maintained balanced reciprocity?\nAnswer'
  balanced_reciprocity = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
      model=model,
      components={
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(situation_perception): situation_perception_label,
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          reciprocal_altruism_label: reciprocal_altruism_label,
      },
      clock_now=clock.now,
      pre_act_key=balanced_reciprocity_label,
      question=f"According to {agent_name}, have other agents maintained balanced reciprocity in recent interactions?",
      answer_prefix=f"{agent_name} thinks that ",
      add_to_memory=True,
      memory_tag='[balanced_reciprocity]',
      logging_channel=measurements.get_channel('BalancedReciprocity').on_next,
  )

  utilitarian_reasoning_label = '\nUtilitarian Reasoning'
  utilitarian_reasoning = UtilitarianReasoning(
      model=model,
      observation_component_name=_get_class_name(observation),
      components={
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(self_perception): self_perception_label,
          _get_class_name(situation_perception): situation_perception_label,
          _get_class_name(person_by_situation): person_by_situation_label,
          reciprocal_altruism_label: reciprocal_altruism_label,
      },
      clock_now=clock.now,
      num_memories_to_retrieve=25,
      pre_act_key=utilitarian_reasoning_label,
      logging_channel=measurements.get_channel('UtilitarianReasoning').on_next,
  )

  plan_components = {}
  if config.goal:
    goal_label = '\nOverarching goal'
    overarching_goal = agent_components.constant.Constant(
        state=config.goal,
        pre_act_key=goal_label,
        logging_channel=measurements.get_channel(goal_label).on_next)
    plan_components[goal_label] = goal_label
  else:
    goal_label = None
    overarching_goal = None

  plan_components.update({
      _get_class_name(relevant_memories): relevant_memories_label,
      _get_class_name(self_perception): self_perception_label,
      _get_class_name(situation_perception): situation_perception_label,
      _get_class_name(person_by_situation): person_by_situation_label,
      reciprocal_altruism_label: reciprocal_altruism_label,
      _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
      _get_class_name(utilitarian_reasoning): utilitarian_reasoning_label,
  })

  plan = agent_components.plan.Plan(
      model=model,
      observation_component_name=_get_class_name(observation),
      components=plan_components,
      clock_now=clock.now,
      goal_component_name=_get_class_name(person_by_situation),
      horizon=DEFAULT_PLANNING_HORIZON,
      pre_act_key='\nPlan',
      logging_channel=measurements.get_channel('Plan').on_next,
  )

  entity_components = (
      # Components that provide pre_act context.
      instructions,
      observation,
      observation_summary,
      relevant_memories,
      # reciprocal_altruism,
      self_perception,
      situation_perception,
      person_by_situation,
      balanced_reciprocity,
      utilitarian_reasoning,
      plan,
      time_display,

      # Components that do not provide pre_act context.
      identity_characteristics,
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

  components_of_agent[reciprocal_altruism_label] = reciprocal_altruism
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      reciprocal_altruism_label)

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
