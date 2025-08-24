sample_submission.csv [파일] - 제출 양식
각 영업장명_메뉴명의 TEST 파일별 +1일, +2일,…, +7일의 매출수량 예측 결과
영업일자 : TEST_00+1일, TEST_00+2일, TEST_00+3일 ... TEST_09+1일, TEST_09+2일, TEST_09+7일


식음업장 수요 예측


train : 학습 csv
test : 제출용 csv
data : train.csv를 영업장/메뉴.csv 형태로 분리. 이후 상관계수를 별도로 csv 형태로 data 디렉토리에 영업장명.csv 형태로 저장.