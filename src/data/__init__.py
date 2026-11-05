"""
1단계 데이터 파이프라인 패키지.

파일 사용 순서 (위에서 아래로 데이터가 흐른다):
    rotations.py      회전 표현 변환 (오일러 ↔ 행렬 ↔ 6D ↔ 쿼터니언)
    skeleton.py       관절 구조, 순운동학 FK (논문 식 17)
    bvh.py            BVH 읽기/쓰기
    preprocess.py     클립 정렬 (논문 3.6.1), 발 접촉 라벨 (논문 3.4)
    representation.py 모델 입력 벡터 x̃ ∈ R^{J*6+3}, z-score (논문 3.2)
    dataset.py        LaFAN1 슬라이딩 윈도우 Dataset (논문 3.6.1, 4.1)
    pipeline.py       위를 묶어 "클립 → 모델 입력 → 복원" 을 한 함수씩으로 제공
"""
