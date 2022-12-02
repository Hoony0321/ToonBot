from static import MODEL_PATH, MODEL_VERSION
import tensorflow as tf
import tensorflow_text as text

reloaded_model = tf.saved_model.load(MODEL_PATH + "/model_{}".format(MODEL_VERSION))

test_text1 = "여자도 군대에 간다면?'본격 여자도 군대 가는 만화!"
test_text2 = "왕은 영웅이 되고 싶어하는 공주의 소원을 들어주기로 전격 결심! 공주를 속이고 마치 영웅이 된 것처럼 만들기 위해 온 나라가 연극을 하게 되는데..파란만장한 그들만의 눈물겨운 영웅만들기의 대장정이 펼쳐집니다~"
test_text3 = "재수생 오설렘은 게임 속에서 만난 이상형 ‘타락엑스칼리’ 에게 푹 빠져 알콩달콩 애정을 쌓던 중…갑작스레 잠수 이별을 당한다.분노의 힘으로 공부를 시작한 설렘은 ‘타락엑스칼리’ 가 다닌다는 명문대 입학에 성공한다!이름도 얼굴도 모르는 ‘타락엑스칼리’ 를 찾아 나서는 설렘.대체 그 녀석은 누굴까? 남주를 서치하는 설렘의 두근두근 로맨스."
test_text4 = "하북팽가 최고의 전력, 도왕 팽지혁. 마교의 습격으로 멸문지화에 빠진 가문을 구하러 낙호곡으로 향하지만 모든 것은 함정이었고, 죽음의 순간 자신이 사랑했던 가문에게 버림받았다는 사실에 분노한다.'다시... 내게 한 번만 더 기회가 주어진다면...!'죽은 줄 알았던 팽지혁은 방구석 둔재, 하북팽가 사공자의 몸에서 눈을 뜬다."

raw_result = reloaded_model([test_text1])
print(raw_result)
print(tf.sigmoid(raw_result))