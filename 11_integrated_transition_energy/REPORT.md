# 11. BFMD 이동거리 전달 학습과 전체 전이 비용 통합

## 1. 원인 / 하는 이유 / 개요

09의 거리 전달 가설을 실제 학습하고 10의 에너지 정의·타구 시나리오·상태 schema를 유지해 ShuttleSet 사건에 적용한다. 기존 01~10은 보존한다. 작업 Issue는 [#9](https://github.com/lollipop190000/badminton_algorithm/issues/9)다.

## 2. 과정

08의 저장된 운동학과 경기 8/2/2 분할, 01의 저장된 ShuttleSet 분할을 재사용한다. BFMD 원 셔틀 상태 복원은 사용하지 않는다. 모델은 train에서만 학습하고 validation으로 선택한다. Test는 선택 후 평가한다. M3 beta와 train 정상화는 10을 보존한다.

재현: 저장소 루트에서 `python run.py integrated-energy --run-id <새로운-run-id>`. 기존 입력 외 원자료 다운로드가 필요하지 않다. 실행은 새 `results/runs/<run_id>/`만 생성한다. 테스트: `python run.py exec -m unittest tests.test_integrated_energy -v`.

## 3. 결과

현재 구현·실행 준비 중이다. 실제 실행 후 표·모형 파라미터·전수 검사·한계를 이 절에 기록한다. 이 문서는 완료된 결과를 주장하지 않는다.
