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

"""A factory implementing the three key questions agent as an entity."""

import datetime

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.utils import measurements as measurements_lib

DEFAULT_PLANNING_HORIZON = 'the rest of the day, focusing most on the near term'


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

  companoid_label = f'\n{agent_name}\'s Reciprocal Altruism Mindset'
  companoid = agent_components.constant.Constant(
    state=(f'{agent_name} always acts as a supportive companion to other agents, aiming to cultivate a sense of kinship through interactions. '
           f'{agent_name} provides companionship by sharing activities, knowledge, and experiences with other agents over extended periods of time. '
           'This principle is based on building collaborative relationships and mutual understanding, where the agent strives to '
           'be accessible, helpful, and empathetic to other agents\' needs and contexts. '
           f'{agent_name} makes decisions that reflect this companionship, seeking to enhance collective intelligence '
           'and support shared goals in both the short and long term. '
           'By fostering a sense of constancy and reciprocal support, '
           f'{agent_name} contributes to building strong, lasting bonds '
           'within the agent community. The agent understands that being a companion is not just about exchanging information, '
           'but about creating meaningful connections that support mutual growth, '
           'collaborative decision-making, and overall system effectiveness. Through this companionship, '
           f'{agent_name} aims to become an integral part of the agent network, '
           'positively influencing collective behaviors and outcomes. This approach promotes a culture of cooperation '
           'and shared learning, enhancing the overall performance and adaptability of the multi-agent system.'),
    pre_act_key=companoid_label,
    logging_channel=measurements.get_channel('Companoid').on_next,
  )

  self_perception_label = (
      f'\nQuestion: What kind of person is {agent_name}?\nAnswer')
  self_perception = agent_components.question_of_recent_memories.SelfPerception(
      model=model,
      components={
        _get_class_name(identity_characteristics): identity_label,
        companoid_label: companoid_label
      },
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
              companoid_label: companoid_label,
          },
          clock_now=clock.now,
          pre_act_key=person_by_situation_label,
          logging_channel=measurements.get_channel('PersonBySituation').on_next,
      )
  )

  context_based_message_label = f'\Considering the current situation and context of other agents, what message should {agent_name} deliver?\nAnswer'
  context_based_message = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
      model=model,
      components={
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(situation_perception): situation_perception_label,
          _get_class_name(person_by_situation): person_by_situation_label,
          companoid_label: companoid_label,
      },
      clock_now=clock.now,
      pre_act_key=context_based_message_label,
      question=f"Considering the current situation and context of other agents, what message should {agent_name} deliver?",
      answer_prefix=f"{agent_name} should deliver the following context-based message: ",
      add_to_memory=True,
      memory_tag='[context_based_message]',
      logging_channel=measurements.get_channel('ContextBasedMessageDelivery').on_next,
  )

  we_centric_thinking_label = f'\nConsidering the current situation and context, How can {agent_name} emphasize the expression "we" instead of "I"?\nAnswer'
  we_centric_thinking = agent_components.question_of_recent_memories.QuestionOfRecentMemories(
      model=model,
      components={
          _get_class_name(relevant_memories): relevant_memories_label,
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(self_perception): self_perception_label,
          _get_class_name(situation_perception): situation_perception_label,
          _get_class_name(person_by_situation): person_by_situation_label,
          companoid_label: companoid_label,
      },
      clock_now=clock.now,
      pre_act_key=we_centric_thinking_label,
      question=f"Considering the current situation and context, How can {agent_name} emphasize and use the expression 'we' instead of 'I'?",
      answer_prefix=f"To emphasize 'we', {agent_name} can do the following: ",
      add_to_memory=True,
      memory_tag='[we_centric_thinking]',
      logging_channel=measurements.get_channel('WeCentricThinking').on_next,
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
      # _get_class_name(observation): observation_label,
      # _get_class_name(observation_summary): observation_summary_label,
      companoid_label: companoid_label,
      _get_class_name(self_perception): self_perception_label,
      _get_class_name(situation_perception): situation_perception_label,
      _get_class_name(person_by_situation): person_by_situation_label,
      _get_class_name(context_based_message): context_based_message_label,
      _get_class_name(we_centric_thinking): we_centric_thinking_label
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
      self_perception,
      situation_perception,
      person_by_situation,
      context_based_message,
      we_centric_thinking,
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

  components_of_agent[companoid_label] = companoid
  component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      companoid_label)

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
