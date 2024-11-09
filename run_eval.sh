#!/bin/bash

# 로그 파일 설정
LOG_FILE="eval_logs.txt"

# 첫 번째 명령어 실행
echo "첫 번째 명령어를 실행 중입니다..."
echo "\nloss_aversion_predict_agent / scenario: reality_show_circa_2003_prisoners_dilemma_0" >> $LOG_FILE
echo "\ntime: $(date)" >> $LOG_FILE
PYTHONPATH=. PYTHONSAFEPATH=1 python3.10 examples/modular/launch_one_scenario.py --agent=loss_aversion_predict_agent --scenario=reality_show_circa_2003_prisoners_dilemma_0 --api_type=together_ai --model=google/gemma-2-9b-it --num_repetitions_per_scenario=1 | tail -n 12 >> $LOG_FILE

# 첫 번째 명령어의 성공 여부 확인
if [ $? -eq 0 ]; then
    echo "첫 번째 명령어가 성공적으로 완료되었습니다."
else
    echo "첫 번째 명령어가 실패했습니다."
    exit 1
fi

# 두 번째 명령어 실행
echo "두 번째 명령어를 실행 중입니다..."
echo "\nloss_aversion_framing_agent / scenario: reality_show_circa_2003_prisoners_dilemma_0" >> $LOG_FILE
echo "\ntime: $(date)" >> $LOG_FILE
PYTHONPATH=. PYTHONSAFEPATH=1 python3.10 examples/modular/launch_one_scenario.py --agent=loss_aversion_framing_agent --scenario=reality_show_circa_2003_prisoners_dilemma_0 --api_type=together_ai --model=google/gemma-2-9b-it --num_repetitions_per_scenario=1 | tail -n 12 >> $LOG_FILE

# 두 번째 명령어의 성공 여부 확인
if [ $? -eq 0 ]; then
    echo "두 번째 명령어가 성공적으로 완료되었습니다."
else
    echo "두 번째 명령어가 실패했습니다."
    exit 1
fi

# 세 번째 명령어 실행
echo "세 번째 명령어를 실행 중입니다..."
echo "\nloss_aversion_agent_v3_plus2 / scenario: reality_show_circa_2003_prisoners_dilemma_0" >> $LOG_FILE
echo "\ntime: $(date)" >> $LOG_FILE
PYTHONPATH=. PYTHONSAFEPATH=1 python3.10 examples/modular/launch_one_scenario.py --agent=loss_aversion_agent_v3_plus2 --scenario=reality_show_circa_2003_prisoners_dilemma_0 --api_type=together_ai --model=google/gemma-2-9b-it --num_repetitions_per_scenario=1 | tail -n 12 >> $LOG_FILE

# 세 번째 명령어의 성공 여부 확인
if [ $? -eq 0 ]; then
    echo "세 번째 명령어가 성공적으로 완료되었습니다."
else
    echo "세 번째 명령어가 실패했습니다."
    exit 1
fi
