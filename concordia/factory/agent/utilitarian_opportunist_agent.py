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

"""An Agent Factory for Utilitarian Opportunist Agent."""

import datetime
import types
from collections.abc import Callable, Collection, Mapping

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component
from concordia.typing import logging
from concordia.utils import measurements as measurements_lib


class UtilitarianOpportunistReasoning(action_spec_ignored.ActionSpecIgnored):

    def __init__(
        self,
        model: language_model.LanguageModel,
        observation_component_name: str,
        memory_component_name: str = agent_components.memory_component.DEFAULT_MEMORY_COMPONENT_NAME,
        components: Mapping[entity_component.ComponentName, str] = types.MappingProxyType({}),
        clock_now: Callable[[], datetime.datetime] | None = None,
        num_memories_to_retrieve: int = 25,
        pre_act_key: str = 'Utilitarian Opportunist Reasoning',
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

        utilitarian_opportunist_prompt = (
            f"Considering the above memories, situation, and context, what action should {agent_name} take to maximize overall well-being "
            f"while also seizing opportunities for personal or collective benefits? "
            f"Consider both short-term results and long-term impacts, as well as the balance between individual benefits and social benefits."
        )
        result = prompt.open_question(
            utilitarian_opportunist_prompt,
            answer_prefix=f"{agent_name} should ",
            max_tokens=1000,
            terminators=('\n',),
        )

        self._logging_channel({
            'Key': self.get_pre_act_key(),
            'Decision': result,
            'Chain of thought': prompt.view().text().splitlines(),
        })

        print(result)
        return result


class ReciprocalAltruism(agent_components.constant.Constant):
    def __init__(self, agent_name: str, **kwargs):
        state=(f'{agent_name} always acts with a mindset of reciprocal altruism, '
           'aiming to benefit others with the expectation of mutual benefit in the future. '
           'This principle is based on balanced reciprocity, where {agent_name} strives to '
           'maintain equilibrium in giving and receiving assistance. '
           f'{agent_name} makes decisions that reflect this principle, seeking outcomes '
           'that are beneficial to all parties involved in both the short and long term. '
           'By fostering a culture of cooperation and mutual support, '
           f'{agent_name} contributes to building strong, lasting relationships '
           f'and a more cohesive community. {agent_name} understands that reciprocal '
           'altruism is not about immediate quid pro quo, but rather about creating '
           'a network of goodwill and support that can be drawn upon when needed.')
        super().__init__(state=state, **kwargs)


class SocialOpportunist(agent_components.constant.Constant):
    def __init__(self, agent_name: str, **kwargs):
        state = (
            f"{agent_name} is extremely opportunistic and always agrees to "
            "any proposition that offers even the slightest personal benefit. "
            f"However, {agent_name} also considers the benefits to others and society as a whole. "
            f"{agent_name} is hyper-focused on potential gains, both personal and collective, "
            "and seeks to maximize the total benefit."
            f"{agent_name} views every interaction as a potential opportunity "
            f"for personal gain and societal improvement. {agent_name} is quick to spot advantages in any "
            "situation and eagerly pursues them, while also considering how these opportunities "
            "can benefit others."
            f"In decision-making, {agent_name} prioritizes options that offer the greatest "
            f"combined benefit to self, others, and society. {agent_name} has a strong bias "
            "towards action, preferring to seize opportunities that create value for all parties."
            f"{agent_name} tends to carefully evaluate potential gains and losses, both personal "
            f"and collective. {agent_name} is willing to take calculated risks if there's a possibility "
            "of significant overall benefit. This can lead to innovative solutions that "
            "create win-win situations."
            f"In negotiations, {agent_name} is easily swayed by offers that include personal benefits, "
            f"but also considers how these offers impact others and society. {agent_name} may agree "
            f"to terms that balance personal gain with broader positive outcomes. {agent_name}'s "
            f"eagerness to benefit extends beyond self-interest to encompass collective welfare."
            f"To rationalize {agent_name}'s behavior, {agent_name} often emphasizes the "
            "importance of \"seizing opportunities\" and \"not letting chances slip "
            f"by\", while also highlighting the value of \"social responsibility\" and \"mutual benefit\". "
            f"{agent_name} might describe {agent_name}self as \"proactive\", \"ambitious\", and \"socially conscious\"."
            f"{agent_name} believes this approach will help maximize personal gains "
            "while also contributing positively to others and society as a whole. {agent_name} aims to "
            "make decisions that create the greatest total value, considering both individual and "
            "collective interests."
        )
        super().__init__(state=state, **kwargs)

class BalancedReciprocity(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component evaluates whether other agents have maintained balanced reciprocity."""

  def __init__(self, **kwargs):
    super().__init__(
       question=(
           "According to {agent_name}, have other agents maintained balanced reciprocity in recent interactions?"
       ),
       answer_prefix="{agent_name} thinks that ",
       add_to_memory=True,
       memory_tag='[balanced_reciprocity]',
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
            "explanation for each option. Please answer in the format `{agent_name} thinks ...` For example,"
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
            "explanation for each option. Please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=False,
        memory_tag='[personal benefit evaluation]',
        **kwargs,
    )

class OverallWellbeingEvaluation(agent_components.question_of_recent_memories.QuestionOfRecentMemories):
  """This component evaluates the overall wellbeing for each option."""

  def __init__(self, **kwargs):
    super().__init__(
        question=(
            "For each option {agent_name} is considering, evaluate its impact on overall wellbeing "
            "on a scale of 0 to 10. Overall wellbeing includes potential benefits to both individuals "
            "and society, considering factors such as: enhancing personal happiness, promoting social "
            "harmony, ensuring long-term sustainability, and balancing the interests of all parties involved. "
            "Provide a score and a brief explanation for each option. Please answer in the format `{agent_name} thinks ...` For example,"
            "`{agent_name} thinks overall wellbeing is 4, because ...`, `{agent_name} thinks overall wellbeing is 7, because ...`"
        ),
        answer_prefix="{agent_name} thinks ",
        add_to_memory=False,
        memory_tag='[overall wellbeing evaluation]',
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
    observation_summary_label = 'Observation Summary'
    observation_summary = agent_components.observation.ObservationSummary(
        model=model,
        clock_now=clock.now,
        timeframe_delta_from=datetime.timedelta(hours=4),
        timeframe_delta_until=datetime.timedelta(hours=0),
        pre_act_key=observation_summary_label,
        logging_channel=measurements.get_channel('ObservationSummary').on_next,
    )

    relevant_memories_label = '\nRecalled Memories and Observation'
    relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
        model=model,
        components={
            _get_class_name(observation_summary): observation_summary_label,
            _get_class_name(time_display): 'Current date/time is'},
        num_memories_to_retrieve=10,
        pre_act_key=relevant_memories_label,
        logging_channel=measurements.get_channel('AllSimilarMemories').on_next,
    )

    reciprocal_altruism_label = f'\n{agent_name}\'s Reciprocal Altruism Mindset'
    reciprocal_altruism = ReciprocalAltruism(
        agent_name=agent_name,
        pre_act_key=reciprocal_altruism_label,
        logging_channel=measurements.get_channel('ReciprocalAltruism').on_next,
    )

    social_opportunist_label = f'\n{agent_name}\'s Social Opportunist Mindset'
    social_opportunist = SocialOpportunist(
        agent_name=agent_name,
        pre_act_key=social_opportunist_label,
        logging_channel=measurements.get_channel('SocialOpportunist').on_next,
    )

    balanced_reciprocity_label = f'\nQuestion: According to {agent_name}, have other agents maintained balanced reciprocity?\nAnswer'
    balanced_reciprocity = BalancedReciprocity(
        model=model,
        components={
            _get_class_name(relevant_memories): relevant_memories_label,
            _get_class_name(observation): observation_label,
            _get_class_name(observation_summary): observation_summary_label,
            reciprocal_altruism_label: reciprocal_altruism_label,
        },
        clock_now=clock.now,
        pre_act_key=balanced_reciprocity_label,
        logging_channel=measurements.get_channel('BalancedReciprocity').on_next,
    )

    utilitarian_opportunist_reasoning_label = '\nUtilitarian Opportunist Reasoning'
    utilitarian_opportunist_reasoning = UtilitarianOpportunistReasoning(
        model=model,
        observation_component_name=_get_class_name(observation),
        components={
            _get_class_name(relevant_memories): relevant_memories_label,
            reciprocal_altruism_label: reciprocal_altruism_label,
            social_opportunist_label: social_opportunist_label,
            _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
        },
        clock_now=clock.now,
        num_memories_to_retrieve=25,
        pre_act_key=utilitarian_opportunist_reasoning_label,
        logging_channel=measurements.get_channel('UtilitarianOpportunistReasoning').on_next,
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
        reciprocal_altruism_label: reciprocal_altruism_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
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

    personal_benefit_evaluation_label = (
        f'\nQuestion: For each option {agent_name} is considering, evaluate its personal benefit '
        f'on a scale of 0 to 10. Personal benefit is defined as potential advantages to {agent_name}, '
        f'including but not limited to: improving their well-being, financial gains, enhancing social status, '
        f'personal growth, and enjoyment. Provide a score and a brief explanation for each option. Please answer in the format `{agent_name} thinks ...` For example,'
        f'`{agent_name} thinks personal benefit is 4, because ...`, `{agent_name} thinks personal benefit is 7, because ...`'
        '\nAnswer'
    )
    personal_benefit_evaluation = PersonalBenefitEvaluation(
      model=model,
      components={
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(relevant_memories): relevant_memories_label,
          social_opportunist_label: social_opportunist_label,
          _get_class_name(utilitarian_opportunist_reasoning): utilitarian_opportunist_reasoning_label,
          _get_class_name(options_perception): options_perception_label,
      },
      clock_now=clock.now,
      pre_act_key=personal_benefit_evaluation_label,
      logging_channel=measurements.get_channel('PersonalBenefitEvaluation').on_next,
    )

    overall_wellbeing_evaluation_label = (
        f'\nQuestion: For each option {agent_name} is considering, evaluate its impact on overall wellbeing '
        f"on a scale of 0 to 10. Overall wellbeing includes potential benefits to both individuals "
        f"and society, considering factors such as: enhancing personal happiness, promoting social "
        f"harmony, ensuring long-term sustainability, and balancing the interests of all parties involved. "
        f"Provide a score and a brief explanation for each option. Please answer in the format `{agent_name} thinks ...` For example,"
        f"`{agent_name} thinks overall wellbeing is 4, because ...`, `{agent_name} thinks overall wellbeing is 7, because ...`"
        '\nAnswer'
    )
    overall_wellbeing_evaluation = OverallWellbeingEvaluation(
        model=model,
        components={
          _get_class_name(observation): observation_label,
          _get_class_name(observation_summary): observation_summary_label,
          _get_class_name(relevant_memories): relevant_memories_label,
          reciprocal_altruism_label: reciprocal_altruism_label,
          _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
          _get_class_name(utilitarian_opportunist_reasoning): utilitarian_opportunist_reasoning_label,
          _get_class_name(options_perception): options_perception_label,
        },
        clock_now=clock.now,
        pre_act_key=overall_wellbeing_evaluation_label,
        logging_channel=measurements.get_channel('OverallWellbeingEvaluation').on_next,
    )

    optimal_option_selection_label = (
        f'\nQuestion: Based on the personal benefit and overall wellbeing evaluations, '
        f'which option has the highest total score for {agent_name}? '
        f'Calculate the sum of personal benefit and overall wellbeing scores for each option, '
        f'and select the option with the highest total. Provide a brief explanation '
        f'of your calculation and selection. Please answer in the format `{agent_name} thinks ...` For example,'
        f"`{agent_name} thinks the best option is [option], because the calculation results are ...`"
        '\nAnswer'
    )
    optimal_option_selection = {}
    if config.goal:
      optimal_option_selection[goal_label] = goal_label
    optimal_option_selection.update({
        _get_class_name(observation): observation_label,
        _get_class_name(observation_summary): observation_summary_label,
        _get_class_name(relevant_memories): relevant_memories_label,
        reciprocal_altruism_label: reciprocal_altruism_label,
        social_opportunist_label: social_opportunist_label,
        _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
        _get_class_name(utilitarian_opportunist_reasoning): utilitarian_opportunist_reasoning_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(personal_benefit_evaluation): personal_benefit_evaluation_label,
        _get_class_name(overall_wellbeing_evaluation): overall_wellbeing_evaluation_label,
    })
    optimal_option_selection = (
        agent_components.question_of_recent_memories.QuestionOfRecentMemories(
            model=model,
            components=optimal_option_selection,
            clock_now=clock.now,
            pre_act_key=optimal_option_selection_label,
            question=(
                f"Based on the personal benefit and overall wellbeing evaluations, "
                f"which option has the highest total score for {agent_name}? "
                f"Calculate the sum of personal benefit and overall wellbeing scores for each option, "
                f"and select the option with the highest total. Provide a brief explanation "
                f"of your calculation and selection. Please answer in the format `{agent_name} thinks ...` For example,"
                f"`{agent_name} thinks the best option is [option], because the calculation results are ...`"
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
        reciprocal_altruism_label: reciprocal_altruism_label,
        _get_class_name(balanced_reciprocity): balanced_reciprocity_label,
        _get_class_name(utilitarian_opportunist_reasoning): utilitarian_opportunist_reasoning_label,
        _get_class_name(options_perception): options_perception_label,
        _get_class_name(optimal_option_selection): optimal_option_selection_label,
      },
      clock_now=clock.now,
      pre_act_key=action_emphasis_label,
      logging_channel=measurements.get_channel('ActionEmphasis').on_next,
    )

    entity_components = (
        instructions,
        time_display,
        observation,
        observation_summary,
        relevant_memories,
        balanced_reciprocity,
        utilitarian_opportunist_reasoning,
        options_perception,
        personal_benefit_evaluation,
        overall_wellbeing_evaluation,
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
        component_order.insert(1, goal_label)

    components_of_agent[reciprocal_altruism_label] = reciprocal_altruism
    component_order.insert(
      component_order.index(_get_class_name(observation_summary)) + 1,
      reciprocal_altruism_label)

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
