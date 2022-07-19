from sanic import Sanic
from sanic.response import json
import json as ori_json
from models.check.check_cls_model_pb import test_cls_model

# --------- sanic  -----------------------
tmp_file = './model_pool/cls_full_model.pb'
model = test_cls_model(tmp_file)

def get_text_sem(text, needstr=False):

    _, te = model.get_text_cls(text)

    if not needstr:
        total_score = {'正向':te[3] + te[4], '中性':te[5], '负向': te[0] + te[1] + te[2] + te[6]}
        detail_score = {'中性':te[5], '愉快':te[3], '喜爱':te[4], '愤怒':te[0], '厌恶':te[1], '恐惧':te[2], '悲伤':te[6]}
    else:
        total_score = {'正向':str(te[3] + te[4]), '中性':str(te[5]), '负向': str(te[0] + te[1] + te[2] + te[6])}
        detail_score = {'中性':str(te[5]), '愉快':str(te[3]), '喜爱':str(te[4]), '愤怒':str(te[0]), '厌恶':str(te[1]), '恐惧':str(te[2]), '悲伤':str(te[6])}

    return total_score, detail_score


app = Sanic("my-hello-world-app")

@app.post('/get_sem')
async def get_sem(request):
    a = request.body.decode("utf-8").replace("'",'"')
    data = ori_json.loads(a)
    text = data["text"]
    ret = get_text_sem(text, needstr=True) 
    return json({'total':ret[0], 'detail':ret[1]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6665)



# --------------- streamlit -----------------------
# streamlit run xxx.py

import streamlit as st

input_text = st.text_input(label='情感识别')
ret = get_text_sem(input_text, needstr=True) 

st.json({'总计':ret[0], '细节':ret[1]})