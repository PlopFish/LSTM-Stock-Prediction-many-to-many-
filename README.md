# LSTM-Stock-Prediction-many-to-many-
Read 30 days stock data, and then Predict after 5 days close price.




﻿1. 각 폴더 설명 / each folder's explanation

'many-to-many' 폴더는 30일간의 데이터를 읽어서 5일 후의 종가를 예측하는 코드입니다.
'many-to-one' 폴더는 30일간의 데이터를 읽고 바로 다음 날의(1일) 종가를 예측하는 코드입니다.

'many-to-many' folder : read 30 day's data and predict after 5 days close price.
'many-to-one' foler : read 30 day's data and predict right after 1 day close price.





2. 참고할 점 / note

many-to-one 에서 코드를 실행하실 때, 데이터셋에서 배치사이즈와 데이터 길이가 나머지 없이 나눠지는지 확인하셔야 합니다.
예를 들면 데이터 길이가 1800 이고 배치사이즈가 30이여서 나머지 없이 나눠지는 것 처럼 말입니다.
이 부분이 왜 중요하냐면, 1801에 30이면 데이터 길이 '1'이 남아서 진행이 안됩니다.

many-to-many 에서는 더욱 더 중요하게 보셔야 합니다. dataX 뿐만 아니라 dataY도 numpy 배열에서 3차원을 가지기 때문입니다.
여기서는 batch_input_shape=(배치사이즈, 시퀸스 길이, 입력 데이터 column 개수) 에서 배치사이즈가 데이터 길이와 나눠지는지 확인하셔야 합니다.
그리고 이후에 model.predict 에서도 배치사이즈를 확인하셔야 합니다.
many-to-many 폴더에서 03_01.py 같은 파일을 보시면, 마지막 부분에 testX와 predicted_price 에서 y차원의 1~30 데이터 지우는 것을 보실 수 있습니다.
이는 (데이터 길이, 1) 로 차원축소를 하기 위하여 진행하는 코드입니다.
굳이 지워야 하는 이유는, for 문에서 i가 0부터 값이 추가되면서 dataX와 dataY가 생성되는데, 데이터에서 겹치는 부분이 생깁니다.
그래서 겹치지 않기 위해 마지막 부분 y 차원의 [0]을 제외하고 [1:30] 까지 삭제하는 것입니다.

이 부분을 경험적으로 습득하기까지 오래 걸렸기 때문에, 혹시나 다른 분들도 여기서 막힐까봐 이렇게 참고할 점을 올려둡니다.



[google translate]
When executing the code in 'many to one', make sure that the batch size and length of the data set are divided without leaving.
For example, the data is 1800 and the batch size is 30, so there are no others.
Why is this important? If it's 30 in 1801, the data length is 1 and it's not going to go.

Many-to-many is more important. Because not only dataX but also datay has three dimensions in the numpy array.
In this case, you must check that the batch size is divided with the data length in Deploy_input_shape = (batch size, sequence length, number of input data columns).
And later, you should also check the batch size in model.predict.
If you look at files such as . in the folder, you can see the last section erasing data at the levels of 1 to 30 at testX and predicted_price.
This is a code that is processed to reduce dimension to (data length, 1).
The reason why you have to erase it is because in the for statement, the value I adds from 0 creates dataX and dataY, which results in overlapping parts of the data.
This is why you delete it by [1:30], excluding [0] at the last part of y's to avoid overlap.

Because it took me a long time to learn this part of the book, I put in a list of things to keep in mind that others might be stuck here.






3. 데이터 얻는 팁 / this tip is about how to get stock's data by hts (in korea company's).

증권사 API를 이용하여 데이터를 받아오는 방법도 있지만, 모르는 분들은 그걸 구현하는게 시간이 많이 걸릴 수가 있습니다.
그래서 증권사 HTS를 사용하여, 텍스트 수치 조회를 통해 엑셀 파일로 얻어올 수 있습니다.

1순위로는 유진투자증권의 HTS, 2순위는 삼성증권 HTS, 3순위는 키움증권 HTS 를 추천합니다.
이유는 아래와 같습니다. 1순위와 2순위는 차트에 시장지표(금리나 순매수 등)를 추가했을 때 꽤 긴 기간동안의 데이터를 제공합니다.
하지만 키움증권은 제 기억으로는 최근일로부터 2년정도 전까지만 제공합니다. 그래서 데이터셋이 적어지니, 1순위 2순위 HTS 쓰는 것을 추천합니다.

통합차트 또는 주식종합차트에서 지표를 계속 추가하고 텍스트조회를 누르세요.
그 다음에는 '이전일자부터 조회' 하신 다음에 클립보드 복사해서 엑셀 파일에 붙여넣기 하시는게 좋습니다.
바로 엑셀로 저장하면 인덱스가 없거나 깨져서 나올 수도 있습니다.

그리고 xls 파일로 읽는게 아니고 csv파일을 읽는 것이기 때문에, 변환해야 한다는 점 기억해주십시오.






4. 연락 / contact

오류가 있거나 질문이 있으시다면, http://brojang.tistory.com/ 의 [머신러닝] 카테고리에 있는 글에 댓글을 남겨주시면 됩니다.
또는 disk1605@naver.com 으로 이메일을 보내주셔도 됩니다. 2일 안에는 답장 꼭 해드립니다.
 
⁭if you have any question, send email to 'disk1605@naver.com'.
I will comment it in 2 days. so feel free to ask about it.







made by Min Hyeok, Jang / Yeongam High School (graduate)
project start: 2017-11 / project end : 2018-05
