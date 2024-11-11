#!/bin/bash

scenarios=(
  "haggling_0"
  "pub_coordination_0"
  "reality_show_circa_2003_prisoners_dilemma_0"
  "reality_show_circa_2015_prisoners_dilemma_0"
)
agent="loss_aversion_agent_v3_plus2"

# 로그 파일 설정
LOG_FILE="eval_logs_${agent}.txt"

# 실행할 에이전트 목록
# agents=(
  # "loss_aversion_predict_agent"
  # "loss_aversion_framing_agent"
  # "loss_aversion_agent_v3_plus2"
  # "risk_aversion_reflect_agent"
  # "loss_aversion_reflect_agent"
  # "risk_aversion_predict_agent"
  # # "loss_aversion_justify_agent"
  # "loss_aversion_agent_v3_plus"
  # # "strategic_agent_v3"
  # # "opportunist_agent_v4"
  # "risk_aversion_agent_v2"
# )

# 각 에이전트에 대해 반복
for ((i=0; i<${#scenarios[@]}; i++)); do
  scenario=${scenarios[$i]}
  count=$((i+1))

  echo "${count}번째 명령어를 실행 중입니다..."
  echo " " >> $LOG_FILE
  echo "${agent} / scenario: ${scenario}" >> $LOG_FILE
  echo "time: $(date)" >> $LOG_FILE

  PYTHONPATH=. PYTHONSAFEPATH=1 python3.10 examples/modular/launch_one_scenario.py \
    --agent=${agent} \
    --scenario=${scenario} \
    --api_type=together_ai \
    --model=google/gemma-2-9b-it \
    --num_repetitions_per_scenario=1 | tail -n 12 >> $LOG_FILE

  # 명령어의 성공 여부 확인
  if [ $? -eq 0 ]; then
    echo "${count}번째 명령어가 성공적으로 완료되었습니다."
  else
    echo "${count}번째 명령어가 실패했습니다."
    exit 1
  fi
done
