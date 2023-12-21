from openai import OpenAI,AsyncOpenAI

import re
import copy
import json
import asyncio
with open('config.json', 'r') as f:
    config_data = json.load(f)

OPENAI_API_KEY = config_data["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
def get_consultation(df, consultation_num):
  return df[df['consultation_num'] == consultation_num]

def get_unique_consultation_nums(df):
    unique_nums = df['consultation_num'].unique()
    return unique_nums


# target_consultation_num 상담 번호 회의록 string return
def create_consultation_history(df, target_consultation_num):
    filtered_df = get_consultation(df, target_consultation_num).sort_values(by='timestamp')
    consultation_history = f"<상담 번호:{target_consultation_num}>\n"
    consultation_history = {}
    consultation_history["id"] = target_consultation_num
    consultation_history["dialogue"] = ""
    for index, row in filtered_df.iterrows():
        if row['txrx'] == 'TX':
            consultation_history["dialogue"] += "상담원: " + str(row['message']) + "\n"
        elif row['txrx'] == 'RX':
            consultation_history["dialogue"] += "고객: " + str(row['message']) + "\n"


    return consultation_history




def get_gpt_summary(consultation):
    if len(re.split(r'[ \n]+', consultation)) >= 4096:
        return "길이 초과"
 
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
                {"role": "system", "content": "You should give a short summary for a consultation"},
        {"role": "user", "content": f"{consultation}\n 고객의 요청과 상담원의 대응을 중심으로 상담 내용을 단 한줄로 요약해줘."},
    ])
    return response.choices[0].message.content


## PREPROCESS FUNCTIONS ##

def delete_newline(dialogue):
  splitted_dialogue = dialogue.split('\n')
  deleted_dialogue = [splitted_dialogue[0]]

  # 같은 사람이 말한 내용 한줄로 압축
  for i in range(1, len(splitted_dialogue)):
    if splitted_dialogue[i-1][0:3] == splitted_dialogue[i][0:3]:
        if splitted_dialogue[i][0:3] == "고객:" :
                deleted_dialogue[-1] += splitted_dialogue[i][3:]
        else:
                deleted_dialogue[-1] += splitted_dialogue[i][4:]
    else:
      deleted_dialogue.append(splitted_dialogue[i])

  return deleted_dialogue



def delete_repeated_word(deleted_dialogue):
  not_repeated_word = ["네", "예", "어", "아"]

  # 의미없이 반복되는 불용어 삭제
  for i in range(0, len(deleted_dialogue)):
    j = 0

    # 불용어만 반복되면 뒤에 반복되는 단어 삭제
    while j < len(deleted_dialogue[i]):
      if deleted_dialogue[i][j] in not_repeated_word:
        deleted_character = False
        for k in range(j+1, len(deleted_dialogue[i])):
          if deleted_dialogue[i][k] not in not_repeated_word and deleted_dialogue[i][k] != " ":
            deleted_character = True
            white_space = " " if deleted_dialogue[i][k-1] == " " else ""
            deleted_dialogue[i] =deleted_dialogue[i][0:j+1] + white_space + deleted_dialogue[i][k:]
            break
        if deleted_character is False:
          deleted_dialogue[i] = deleted_dialogue[i][0:j+1]
          break
      j += 1

  return '\n'.join(deleted_dialogue)


def get_biased_dialogues_removed_list(dialogue_list,ratio = 0.25):
  def is_biased(dialogue):
    num_cons = dialogue.count("상담원")
    num_cus = dialogue.count("고객")
    num_total = num_cons + num_cus
    if (num_cons / num_total) < ratio or (num_cus / num_total) < ratio:
      return False
    return True

  return[dialogue for dialogue in dialogue_list if is_biased(dialogue['dialogue'])]

def get_duplicated_removed_list(data):
  now_data = copy.deepcopy(data)

  for idx1, i in enumerate(data):
    for idx2, j in enumerate(data):
      if idx1 <= idx2:
        continue
      if i["dialogue"] == j["dialogue"]:
        now_data.pop(idx1)

  return now_data
  


def preprocess_data(data):
  preprocessed_data = copy.deepcopy(data)

  for i in range(0, len(preprocessed_data)):
    preprocessed_data[i]['dialogue'] = delete_newline(preprocessed_data[i]['dialogue'])
    preprocessed_data[i]['dialogue'] = delete_repeated_word(preprocessed_data[i]['dialogue'])

  preprocessed_data = get_biased_dialogues_removed_list(preprocessed_data)
  preprocessed_data = get_duplicated_removed_list(preprocessed_data)
  return preprocessed_data


def postprocess_data(data):
  postprocessed_data = []

  for i in data:
    if i['summary'] > config_data["min_summary_length"] and i['summary'] < config_data["max_summary_length"]:
       postprocessed_data.append(i)

  return postprocessed_data

   