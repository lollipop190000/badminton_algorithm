# 확인한 문헌과 적용 한계

확인일: 2026-10-14. 숫자 추출과 원문 접근 상태를 구별한다. 원문 PDF를 이 저장소에 재배포하지 않는다.

| 자료 | 확인한 내용 | 이 구현에서의 사용과 한계 |
|---|---|---|
| Pugliese et al. (2025), *Energy Cost of Badminton Footwork: A Novel Experimental Approach*, [DOI](https://doi.org/10.1123/ijspp.2024-0267) | 08에 보존된 publisher Crossref 초록의 19.59/20.38 J/(kg·m), 발행 정보 및 공식 초록 링크를 재확인. Publisher 본문/시행별 자료 접근은 미확보 | 산술평균 19.985를 M0의 외부 benchmark로 유지. 경기 개별 비용으로 보정된 값이 아님 |
| Zago et al. (2018), *Kinematic algorithm to determine the energy cost of running with changes of direction*, [DOI](https://doi.org/10.1016/j.jbiomech.2018.05.028), [저자 저장소](https://air.unimi.it/handle/2434/578932) | 26명, 5m shuttle run, 3D 운동학·지면 접촉·무릎 굽힘을 사용하는 제동 분리. 공개 accepted PDF는 403으로 미열람 | 가감속 분리의 근거. BFMD 2D ground proxy를 그 검증된 C_est로 부르지 않으며 원식/효율을 검증했다고 주장하지 않음 |
| Taboga & Kram (2019), *Modelling the effect of curves on distance running performance*, [DOI](https://doi.org/10.7717/peerj.8222) | 공개 PDF의 식 (1)/(8)에서 axial-force 증가와 0.6234 계수 확인. 원문 식 텍스트는 검색 인덱스에서 열람, 직접 PDF URL은 403 | M3의 sqrt(1+(a_perp/g)²)-1 모양과 beta_c의 가설 근거. 지속 곡선 달리기의 결과를 급정지 풋워크 계수로 검증했다고 하지 않음 |
| Riddick & Kuo (2022), *Mechanical work accounts for most of the energetic cost in human running*, [DOI](https://doi.org/10.1038/s41598-021-04215-6) | 연구 본문에서 근육/힘줄 일과 대사 변환 가정의 범위 및 불확실성 확인 | 수평 속도 변화가 전체 근육 일 또는 대사비용과 같지 않다는 한계. beta_+=1, beta_-=0.25는 이 연구의 피팅 결과가 아님 |
| Hatamoto et al. (2014), *The Relationship between Running Velocity and the Energy Cost of Turning during Running*, [DOI](https://doi.org/10.1371/journal.pone.0081850) | PLOS 원문: 9명, 180° 방향전환, 속력·전환 빈도에 따른 산소소비. Turn 비용은 감속/재가속까지 포함 | 방향전환과 X±를 독립적인 실측 부담으로 더하면 중복될 수 있음을 검토. 논문의 회귀식을 Xc 계수로 직접 대입하지 않음 |
| Towler et al. (2023), *Effects of racket moment of inertia on racket head speed, impact location and shuttlecock speed during the badminton smash*, [DOI](https://doi.org/10.1038/s41598-023-37108-x) | Nature 원문: 20명, 관성모멘트 85–106 kg·cm², 체중 68.6±8.6kg, head speed 52.0±5.3m/s, 관측 범위 42.2–63.3m/s | I=0.00955kg·m²와 체중 기준 및 스매시 고속 시나리오. 대사 비용 실험이 아니며 grip translation·shaft flexibility를 고정축 모델이 완전히 표현하지 못함 |
| Phomsoupha, Ibrahime & Laffaye (2024), *Badminton racket deflection, comparison between rigid versus flexible according to different strokes*, [DOI](https://doi.org/10.30827/ijrss.33203), [대학 보관 PDF](https://digibug.ugr.es/bitstream/handle/10481/102321/128-Article%20Text-1982-1-10-20241126.pdf?isAllowed=y&sequence=1) | 7쪽 PDF를 실제 다운로드하고 p.35 Table 1/2를 렌더링 확인. 8명, 500Hz, head speed drop 6.3±1.9, clear 20.9±4.1, smash 34.6±3.4, full smash 49.2±5.1m/s | 3타구의 속도 입력과 나머지 5타구 pooled prior의 공개 수치. 5종에 각각 직접 대응하는 측정자료가 없으며 pooled 값에 경험적 타구 서열을 부여하지 않음 |
| Wong, Cluff & Kuo (**2021**), *The energetic basis for smooth human arm movements*, [DOI](https://doi.org/10.7554/eLife.68013), [공개 본문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8741215/) | 원문 발행은 2021-12-20이며 요청문 2022는 정정. 10명의 cyclic bimanual reaches, force-rate 비용과 기계적 일을 구별 | 라켓 운동에너지만으로 스윙 대사비용을 식별할 수 없다는 근거. reaching 결과를 badminton swing에 직접 환산하지 않음 |
| Lee & Kim (2023), *Muscle Activation Patterns of Upper Extremity During Different Strokes of Badminton: A Single Subject Pilot Study*, [저자 발표](https://commons.nmu.edu/isbs/vol41/iss1/71/) | 공식 proceedings 초록: recreational player 1명의 drive/clear/smash/drop EMG 비교 | EMG의 타구 차이만으로 J/kg 또는 8종 에너지 순서를 만들지 않음 |

BFMD는 [공식 저장소](https://github.com/Ning-D/BFMD)의 현재 문서와 08의 고정 다운로드 감사에서 source ID/해시를 확인했다. 실제 96개 주석은 이번 실행에서 모두 해시 검증하고 셔틀·hit·court·bbox·pose·타구 timeline을 읽었다. 원 `shot_type` timeline의 ranges는 프레임이며 hit_inferred와 동일 프레임인 것을 실제 파일에서 확인했다. 단일 카메라의 3D 셔틀 높이는 주어지지 않는다.

첨부된 Galeano et al. (2022), [DOI](https://doi.org/10.1016/j.chaos.2022.112828) 논문은 3타구 패턴과 흡수 상태 EPV의 연구 배경으로 읽었다. 이 저장소는 01의 셔틀 도착 구획과 04 parity를 이미 채택했으므로 원 논문의 선수 위치 패턴과 완전히 같은 관측변수라고 주장하지 않는다.

기계적 일 시나리오에서 `(eta_positive,|eta_negative|)=(0.25,1.2)`, lever=0.5m, upper-body multiplier=1은 연구팀이 검토할 가정이다. 하한/상한의 efficiency·lever·체중·미관측 부담 범위 역시 지정 시나리오이며 실제 신뢰구간/상한으로 검증되지 않았다. M3와 C_shot은 대사 정답을 확보하기 전에는 정책·피로의 물리적 결론에 사용할 수 없다.
