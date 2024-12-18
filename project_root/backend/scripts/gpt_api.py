import openai
import configparser
import pandas as pd  # generate_report 関数で使用するため追加

def load_gpt_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    if 'openai' not in config.sections():
        print(f"設定ファイル '{config_file}' に [openai] セクションがありません。")
        return None

    openai_api_key = config.get('openai', 'api_key', fallback='')
    return openai_api_key

def get_openai_api_key():
    api_key = load_gpt_config()
    if api_key is None or not api_key:
        raise ValueError("OpenAI API キーが取得できませんでした。config.iniを確認してください。")
    return api_key

def check_gpt_api(model="gpt-4o-mini"):
    """
    GPT APIへの接続確認を行います。
    簡単なメッセージを送信し、正常に応答が返ってくるかを確認します。
    """
    openai.api_key = get_openai_api_key()
    messages = [
        {"role": "system", "content": "あなたは確認を行うアシスタントです。"},
        {"role": "user", "content": "GPT APIが正常に動作していますか？"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        reply = response.choices[0].message['content'].strip()
        if reply:
            print("GPT API 接続確認: 成功")
            return True
        else:
            print("GPT API 接続確認: 応答が空です。")
            return False
    except openai.OpenAIError as e:
        print(f"GPT API 接続確認エラー: {e}")
        return False