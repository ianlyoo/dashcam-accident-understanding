# Data and licences

데이터, 프레임, 테스트 ID 목록, 모델 가중치는 이 저장소에 포함하지 않는다. 아래 링크에서 각 사용자가 직접 조건을 확인하고 취득해야 한다. `DATA_DIR`은 사용자가 지정하는 로컬 데이터 디렉터리다. 외부 데이터의 조건은 이 프로젝트 코드의 MIT 라이선스로 바뀌지 않는다.

| 자료 | 출처·취득 방법 | 기록된 조건 | 사용 범위 |
|---|---|---|---|
| DACON 236753 공개 학습/예시 | [대회 데이터 페이지](https://dacon.io/competitions/official/236753/data)에서 대회 규정에 따라 받기 | 대회 제공; 독립적인 재배포 허가는 확인되지 않음 | 세 단계의 계약·공개 예시와 일부 보정. 숨은 시험은 학습에 쓰지 않음 |
| Scoop recapture attack | [Zenodo 16748852](https://zenodo.org/records/16748852)에서 직접 받기 | CC BY 4.0 | S1 원본/재촬영 학습과 기하 특징 평가; 최종 S1 FIT 194개 |
| DLC-2021 | [Zenodo 6466768](https://zenodo.org/records/6466768)에서 원본 아카이브 받기 | CC BY-SA 2.5 Generic; Generated Photos 표기 요청 | S1 재촬영 학습; 최종 FIT 288개. 초기에 거절된 파일럿에도 사용 |
| comma2k19 | [comma.ai 저장소](https://github.com/commaai/comma2k19) 안내에 따라 Chunk 자료 받기 | 저장소의 MIT 고지 유지; 개별 영상·차량 신호 사용 조건 확인 | S3 CAN 신호에서 가감속·조향 프록시 목표를 만들고 모델 학습·검증 |
| Nexar Collision Prediction | [Hugging Face 데이터셋](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction)에서 접근 권한을 얻어 직접 받기; 원 실험은 revision `aa97deda5a59f00bb7187739053b7c72e14374df` | Nexar Open Data License: 출처 표기, 고지 유지, 영리 재판매·재허락 제한 및 윤리 조건. 세부 원문은 데이터셋 페이지 참조 | S2 긴 영상 충돌 후보 학습·OOF 평가(양성 750개). **데이터와 프레임은 절대 재배포하지 않음** |
| Car Crash Dataset (CCD) | [원 프로젝트](https://github.com/Cogito2012/CarCrashDataset); 당시 사용한 [Kaggle 미러](https://www.kaggle.com/datasets/lelesaad/carcrashdataset)에서 별도 취득 | 코드 저장소·미러는 MIT로 표기되지만 영상은 YouTube 유래로 별도 권리 확인 필요 | S2 규칙의 비교 및 AI 프록시 라벨 검토. 최종 충돌 랭커의 선택된 학습 자료는 아님. **영상·프레임 재배포 금지** |
| Sama Drives California | [Hugging Face 데이터셋](https://huggingface.co/datasets/SamaAI/sama-drives-california)에서 직접 받기 | CC BY 4.0, Sama 표기 | 차선·차량 추적 파일럿과 S114 진입 모델; 최종 S182의 새 학습 자료는 아님 |
| UniqueData 원본·재촬영 | 원 제공처의 [공개 자료](https://huggingface.co/datasets/UniqueData)와 조건을 별도 확인 | 당시 기록은 CC BY-NC-ND 4.0; 대회 수상 목적 적격성 미확인 | 자격 검토용 반입만 수행, 점수 제출용 학습에는 사용하지 않음 |
| 사람이 검토한 DKB·Jungmin 소형 패널 | 프로젝트 내부 평가 패널; 외부 배포 링크 없음 | 제작자와 재배포 권리 미확인 | S176·S186 규칙 게이트의 소형 검토. 패널 내용은 미포함 |

최종 S1의 기록된 FIT 구성은 Scoop 194, DLC 288, DACON 공개 TRAIN 10개다. 모델 평가용으로 분리한 76개는 FIT에서 제외했다. CCD와 Nexar 위의 AI 주석은 과제의 정답이 아니라 프록시다. Nexar의 `time_of_event`도 실제 접촉 시각과 같다고 보장되지 않는다.

## 사전학습 모델과 코드

| 자산 | 출처 | 기록된 조건·역할 |
|---|---|---|
| DINOv2 Small/Base/Large | [Meta DINOv2](https://github.com/facebookresearch/dinov2), [Hugging Face 모델](https://huggingface.co/facebook/dinov2-large) | Apache-2.0; S1 특징 및 S2 시각 국소화. 가중치 미포함 |
| V-JEPA 2.1 | [Meta V-JEPA](https://github.com/facebookresearch/vjepa2) | 원 코드 MIT 및 배포물의 Apache 고지 확인; S3 영상 특징. 가중치 미포함 |
| TorchVision 검출기 | [TorchVision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause 코드. COCO 주석 CC BY 4.0, 원 이미지별 권리 별도. S2 박스 특징 |
| DACON 베이스라인 | [대회 자료](https://dacon.io/competitions/official/236753/data) | 원문 조건 확인 필요; 이 저장소는 자체 점수 계산 코드와 최종 추론 소스만 보존 |

제출물에서 추출한 외부 고지는 `src/final_pipeline/model/licenses/`에 남겼다. 권리 문의가 있었던 자료는 공개 저장소에서 제외했다.
