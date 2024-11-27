from openai import OpenAI
import open_clip
from transformers import AutoTokenizer, GPT2Tokenizer


client = OpenAI(
    base_url='xxxx',
    api_key='xxxx',
)

system_content = ("Write a comprehensive text summary of the user’s app usage activities,"
                  "including their preferences, intents, and demands based on the sequence of apps used,"
                  "while refraining from mentioning specific numerical data."
                  "The app sequence format：[session start time, session end time, location functional property,"
                  "app sequence], where the format of app sequence is "
                  "[app1(generated Traffic/byte), app2(generated Traffic/byte),...]")

# sample
user_content = "app sequence [11:09:54, 11:15:18, Restaurant, [Wechat(2583), DaZhongDianPing(1659), AppleMap(10342), Alipay(3124)]]"

chat = client.chat.completions.create(
    messages=[
        {"role": "system",
         "content": system_content,
         },
        {"role": "user",
         "content": user_content,
         }
    ],
    model="gpt-4o",
    max_tokens=100,
    temperature=1,
)

msg = chat.choices[0].message.content
# print(msg)
