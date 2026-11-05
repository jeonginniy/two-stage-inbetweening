"""
two_stage_inbetweening 의 라이브러리 코드 패키지.

"Motion In-betweening via Two-stage Transformers" (Qin et al., ACM TOG 2022) 를
직접 구현하는 프로젝트. 프로젝트 루트에서 `import src.data...` 형태로 사용한다.

- src/config.py : configs/ YAML 로더 (경로, device)
- src/data/     : 1단계 데이터 파이프라인 (BVH → 모델 입력 → 복원)
- (2단계 이후) src/models/, src/training/ 등이 추가될 예정
"""
